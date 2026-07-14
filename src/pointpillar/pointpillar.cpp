/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "pointpillar.hpp"

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"
#include "common/tensor.hpp"

namespace pointpillar {
namespace lidar {

class CoreImplement: public Core {
public:
    virtual ~CoreImplement() {
        if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
    }

    bool init(const CoreParameter& param) {
        lidar_voxelization_ = create_voxelization(param.voxelization);
        if (lidar_voxelization_ == nullptr) {
            printf("Failed to create lidar voxelization.\n");
            return false;
        }

        checkRuntime(cudaDeviceSynchronize());
        printf("Loaded lidar voxelization.\n");

        lidar_backbone_ = create_backbone(param.lidar_model);
            if (lidar_backbone_ == nullptr) {
            printf("Failed to create lidar backbone & head.\n");
            return false;
        }

        checkRuntime(cudaDeviceSynchronize());
        printf("Loaded lidar backbone & head.\n");

        PostProcessParameter postprocess_param = param.lidar_post;
        const nvtype::Int2 model_feature_size = lidar_backbone_->feature_size();
        postprocess_param.feature_size = model_feature_size;
        const int feature_cells = postprocess_param.feature_size.x * postprocess_param.feature_size.y;
        const int expected_cls = feature_cells * postprocess_param.num_anchors * postprocess_param.num_classes;
        const int expected_box = feature_cells * postprocess_param.num_anchors * postprocess_param.num_box_values;
        const int expected_dir = feature_cells * postprocess_param.num_anchors * 2;

        printf(
            "[PointPillar] backbone output vs. config:\n"
            "  feature_size (W x H): %d x %d (%d cells)\n"
            "  num_classes=%d, num_anchors=%d, num_box_values=%d\n"
            "  cls: %d (expected %d)  box: %d (expected %d)  dir: %d (expected %d)\n",
            postprocess_param.feature_size.x, postprocess_param.feature_size.y, feature_cells,
            postprocess_param.num_classes, postprocess_param.num_anchors, postprocess_param.num_box_values,
            lidar_backbone_->cls_numel(), expected_cls,
            lidar_backbone_->box_numel(), expected_box,
            lidar_backbone_->dir_numel(), expected_dir);

        if (lidar_backbone_->cls_numel() != expected_cls ||
            lidar_backbone_->box_numel() != expected_box ||
            lidar_backbone_->dir_numel() != expected_dir) {
            printf(
                "PointPillar output/config mismatch: cls %d (expected %d), box %d (expected %d), dir %d (expected %d).\n",
                lidar_backbone_->cls_numel(), expected_cls,
                lidar_backbone_->box_numel(), expected_box,
                lidar_backbone_->dir_numel(), expected_dir);
            return false;
        }

        lidar_postprocess_ = create_postprocess(postprocess_param);
        if (lidar_postprocess_ == nullptr) {
            printf("Failed to create lidar postprocess.\n");
            return false;
        }

        printf("Loaded lidar postprocess.\n");

        capacity_points_ = static_cast<size_t>(param.voxelization.max_points);
        bytes_capacity_points_ = capacity_points_ * param.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
        param_ = param;
        return true;
    }

    std::vector<BoundingBox> forward_only(const float *lidar_points, int num_points, void *stream) {
        int cappoints = static_cast<int>(capacity_points_);
        if (num_points > cappoints) {
            printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
        }

        num_points = std::min(cappoints, num_points);

        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        size_t bytes_points = num_points * param_.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points, bytes_points, cudaMemcpyDefault, _stream));

        this->lidar_voxelization_->forward(lidar_points_device_, num_points, _stream);
        this->lidar_backbone_->forward(this->lidar_voxelization_->features(), this->lidar_voxelization_->coords(), this->lidar_voxelization_->params(), _stream);
        this->lidar_postprocess_->forward(this->lidar_backbone_->cls(), this->lidar_backbone_->box(), this->lidar_backbone_->dir(), _stream);

        return this->lidar_postprocess_->bndBoxVec();
    }

    std::vector<BoundingBox> forward_timer(const float *lidar_points, int num_points, void *stream) {
        int cappoints = static_cast<int>(capacity_points_);
        if (num_points > cappoints) {
            printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
        }

        num_points = std::min(cappoints, num_points);

        printf("==================PointPillars===================\n");
        std::vector<float> times;
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        timer_.start(_stream);

        size_t bytes_points = num_points * param_.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points, bytes_points, cudaMemcpyDefault, _stream));
        timer_.stop("[NoSt] CopyLidar");

        timer_.start(_stream);
        this->lidar_voxelization_->forward(lidar_points_device_, num_points, _stream);
        times.emplace_back(timer_.stop("Lidar Voxelization"));

        timer_.start(_stream);
        this->lidar_backbone_->forward(this->lidar_voxelization_->features(), this->lidar_voxelization_->coords(), this->lidar_voxelization_->params(), _stream);
        times.emplace_back(timer_.stop("Lidar Backbone & Head"));

        timer_.start(_stream);
        this->lidar_postprocess_->forward(this->lidar_backbone_->cls(), this->lidar_backbone_->box(), this->lidar_backbone_->dir(), _stream);
        times.emplace_back(timer_.stop("Lidar Decoder + NMS"));

        float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
        printf("Total: %.3f ms\n", total_time);
        printf("=============================================\n");
        return this->lidar_postprocess_->bndBoxVec();
    }

    virtual std::vector<BoundingBox> forward(const float *lidar_points, int num_points, void *stream) override {
        if (enable_timer_) {
            return this->forward_timer(lidar_points, num_points, stream);
        } else {
            return this->forward_only(lidar_points, num_points, stream);
        }
    }

    virtual void set_timer(bool enable) override { enable_timer_ = enable; }

    virtual void print() override {
        lidar_backbone_->print();
    }

private:
    CoreParameter param_;
    nv::EventTimer timer_;
    float* lidar_points_device_ = nullptr;
    size_t capacity_points_ = 0;
    size_t bytes_capacity_points_ = 0;

    std::shared_ptr<Voxelization> lidar_voxelization_;
    std::shared_ptr<Backbone> lidar_backbone_;
    std::shared_ptr<PostProcess> lidar_postprocess_;

    bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar
