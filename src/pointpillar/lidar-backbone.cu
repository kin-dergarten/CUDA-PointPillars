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

#include <cuda_fp16.h>

#include <numeric>

#include "lidar-backbone.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

#include <iostream>
#include "stdio.h"

namespace pointpillar {
namespace lidar {

class BackboneImplement : public Backbone {
public:
    virtual ~BackboneImplement() {
        if (workspace_) checkRuntime(cudaFree(workspace_));
    }

    bool init(const std::string& model) {        
        engine_ = TensorRT::load(model);
        if (engine_ == nullptr) return false;

        cls_dims_ = engine_->static_dims(3);
        box_dims_ = engine_->static_dims(4);
        dir_dims_ = engine_->static_dims(5);

        auto align256 = [](size_t s) { return (s + 255) & ~size_t(255); };
        auto inMB = [](size_t s) { return s / 1024.0 / 1024.0; };

        const size_t cls_sz = align256(
            static_cast<size_t>(std::accumulate(cls_dims_.begin(), cls_dims_.end(), 1, std::multiplies<int32_t>())) * sizeof(float));
        const size_t box_sz = align256(
            static_cast<size_t>(std::accumulate(box_dims_.begin(), box_dims_.end(), 1, std::multiplies<int32_t>())) * sizeof(float));
        const size_t dir_sz = align256(
            static_cast<size_t>(std::accumulate(dir_dims_.begin(), dir_dims_.end(), 1, std::multiplies<int32_t>())) * sizeof(float));

        checkRuntime(cudaMalloc(&workspace_, cls_sz + box_sz + dir_sz));
        uint8_t* base = static_cast<uint8_t*>(workspace_);
        cls_ = reinterpret_cast<float*>(base); 
        base += cls_sz;
        box_ = reinterpret_cast<float*>(base); 
        base += box_sz;
        dir_ = reinterpret_cast<float*>(base);

        std::cout << "Lidar Backbone Memory Usage (MB) and Addresses:" << std::endl;
        std::cout << "  cls:        " << inMB(cls_sz) << " MB\t@ " << static_cast<const void*>(cls_) << std::endl;
        std::cout << "  box:        " << inMB(box_sz) << " MB\t@ " << static_cast<const void*>(box_) << std::endl;
        std::cout << "  dir:        " << inMB(dir_sz) << " MB\t@ " << static_cast<const void*>(dir_) << std::endl;
        std::cout << "  WORKSPACE:  " << inMB(cls_sz + box_sz + dir_sz) << " MB\t@ " << static_cast<const void*>(workspace_) << std::endl;
        
        std::cout << "Lidar Backbone Engine Details:" << std::endl;
        this->print();
        std::cout << std::endl;

        return true;
    }

    virtual void print() override { engine_->print("Lidar Backbone"); }

    virtual void forward(const nvtype::half* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream = nullptr) override {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
        engine_->forward({voxels, voxel_idxs, params, cls_, box_, dir_}, static_cast<cudaStream_t>(_stream));
    }

    virtual float* cls() override { return cls_; }
    virtual float* box() override { return box_; }
    virtual float* dir() override { return dir_; }
    virtual int cls_numel() const override { return numel(cls_dims_); }
    virtual int box_numel() const override { return numel(box_dims_); }
    virtual int dir_numel() const override { return numel(dir_dims_); }
    virtual nvtype::Int2 feature_size() const override {
        // The PointPillar head exports NHWC tensors: [1, H, W, C].
        return nvtype::Int2(cls_dims_[2], cls_dims_[1]);
    }

private:
    static int numel(const std::vector<int>& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    }

    std::shared_ptr<TensorRT::Engine> engine_;
    void*  workspace_ = nullptr;
    float *cls_ = nullptr;
    float *box_ = nullptr;
    float *dir_ = nullptr;
    std::vector<int> cls_dims_, box_dims_, dir_dims_;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
  std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar