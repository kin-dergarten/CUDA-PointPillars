/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: MIT
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

#include "tensorrt.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferVersion.h"
#include "check.hpp"

namespace TensorRT {

static class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
} gLogger_;

static std::string format_shape(const nvinfer1::Dims &shape) {
  char buf[200] = {0};
  char *p = buf;
  for (int i = 0; i < shape.nbDims; ++i) {
    if (i + 1 < shape.nbDims)
      p += sprintf(p, "%d x ", shape.d[i]);
    else
      p += sprintf(p, "%d", shape.d[i]);
  }
  return buf;
}

static const char *data_type_string(nvinfer1::DataType dt) {
  switch (dt) {
  case nvinfer1::DataType::kFLOAT:
    return "Float32";
  case nvinfer1::DataType::kHALF:
    return "Float16";
  case nvinfer1::DataType::kINT32:
    return "Int32";
  // case nvinfer1::DataType::kUINT8: return "UInt8";
  case nvinfer1::DataType::kINT8:
    return "Int8";
  case nvinfer1::DataType::kBOOL:
    return "BOOL";
  default:
    return "Unknow";
  }
}

template <typename _T> static void destroy_pointer(_T *ptr) {
  if (ptr) {
#if NV_TENSORRT_MAJOR >= 10
    delete ptr;
#else
    ptr->destroy();
#endif
  }
}

class __native_engine_context {
public:
  virtual ~__native_engine_context() { destroy(); }

  bool construct(const void *pdata, size_t size, const char *message_name) {
    destroy();

    if (pdata == nullptr || size == 0) {
      printf("Construct for empty data found.\n");
      return false;
    }

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(gLogger_),
        destroy_pointer<nvinfer1::IRuntime>);
    if (runtime_ == nullptr) {
      printf("Failed to create tensorRT runtime: %s.\n", message_name);
      return false;
    }

#if NV_TENSORRT_MAJOR >= 10
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(pdata, size),
#else
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(pdata, size, nullptr),
#endif
        destroy_pointer<nvinfer1::ICudaEngine>);
    if (engine_ == nullptr) {
      printf("Failed to deserialize engine: %s\n", message_name);
      return false;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        destroy_pointer<nvinfer1::IExecutionContext>);
    if (context_ == nullptr) {
      printf("Failed to create execution context: %s\n", message_name);
      return false;
    }
    return context_ != nullptr;
  }

private:
  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

public:
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
};

class EngineImplement : public Engine {
public:
  std::shared_ptr<__native_engine_context> context_;
  std::unordered_map<std::string, int> binding_name_to_index_;

  virtual ~EngineImplement() = default;

  bool construct(const void *data, size_t size, const char *message_name) {
    context_ = std::make_shared<__native_engine_context>();
    if (!context_->construct(data, size, message_name)) {
      return false;
    }

    setup();
    return true;
  }

  bool load(const std::string &file) {
    int fd = ::open(file.c_str(), O_RDONLY);
    struct stat st;
    if (fd < 0 || ::fstat(fd, &st) != 0 || st.st_size == 0) {
      printf(
          "An empty file has been loaded. Please confirm your file path: %s\n",
          file.c_str());
      if (fd >= 0)
        ::close(fd);
      return false;
    }
    void *mapped = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                          MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
      printf("Failed to mmap engine file: %s\n", file.c_str());
      return false;
    }
    ::madvise(mapped, static_cast<size_t>(st.st_size), MADV_SEQUENTIAL);
    bool ok =
        this->construct(mapped, static_cast<size_t>(st.st_size), file.c_str());
    ::munmap(mapped, static_cast<size_t>(st.st_size));
    return ok;
  }

  void setup() {
    auto engine = this->context_->engine_;
#if NV_TENSORRT_MAJOR >= 10
    tensor_names_.clear();
    binding_name_to_index_.clear();
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      tensor_names_.emplace_back(engine->getIOTensorName(i));
      binding_name_to_index_[tensor_names_.back()] = i;
    }
#else
    int nbBindings = engine->getNbBindings();

    binding_name_to_index_.clear();
    for (int i = 0; i < nbBindings; ++i) {
      const char *bindingName = engine->getBindingName(i);
      binding_name_to_index_[bindingName] = i;
    }
#endif
  }

#if NV_TENSORRT_MAJOR >= 10
  const char *tensor_name(int index) const {
    Assertf(index >= 0 && index < static_cast<int>(tensor_names_.size()),
            "Invalid TensorRT tensor index: %d", index);
    return tensor_names_[index].c_str();
  }
#endif

  virtual int index(const std::string &name) override {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(),
            "Can not found the binding name: %s", name.c_str());
    return iter->second;
  }

  virtual bool forward(const std::vector<const void *> &bindings, void *stream,
                       void *input_consum_event) override {
#if NV_TENSORRT_MAJOR >= 10
    if (bindings.size() != tensor_names_.size())
      return false;
    for (size_t i = 0; i < bindings.size(); ++i) {
      if (!context_->context_->setTensorAddress(
              tensor_name(static_cast<int>(i)),
              const_cast<void *>(bindings[i]))) {
        return false;
      }
    }
    (void)input_consum_event;
    return context_->context_->enqueueV3(static_cast<cudaStream_t>(stream));
#else
    return this->context_->context_->enqueueV2(
        (void **)bindings.data(), (cudaStream_t)stream,
        (cudaEvent_t *)input_consum_event);
#endif
  }

  virtual std::vector<int> run_dims(const std::string &name) override {
    return run_dims(index(name));
  }

  virtual std::vector<int> run_dims(int ibinding) override {
#if NV_TENSORRT_MAJOR >= 10
    auto dim = context_->context_->getTensorShape(tensor_name(ibinding));
#else
    auto dim = this->context_->context_->getBindingDimensions(ibinding);
#endif
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<int> static_dims(const std::string &name) override {
    return static_dims(index(name));
  }

  virtual std::vector<int> static_dims(int ibinding) override {
#if NV_TENSORRT_MAJOR >= 10
    auto dim = context_->engine_->getTensorShape(tensor_name(ibinding));
#else
    auto dim = this->context_->engine_->getBindingDimensions(ibinding);
#endif
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual int num_bindings() override {
#if NV_TENSORRT_MAJOR >= 10
    return static_cast<int>(tensor_names_.size());
#else
    return this->context_->engine_->getNbBindings();
#endif
  }

  virtual bool is_input(int ibinding) override {
#if NV_TENSORRT_MAJOR >= 10
    return context_->engine_->getTensorIOMode(tensor_name(ibinding)) ==
           nvinfer1::TensorIOMode::kINPUT;
#else
    return this->context_->engine_->bindingIsInput(ibinding);
#endif
  }

  virtual bool set_run_dims(const std::string &name,
                            const std::vector<int> &dims) override {
    return this->set_run_dims(index(name), dims);
  }

  virtual bool set_run_dims(int ibinding,
                            const std::vector<int> &dims) override {
    nvinfer1::Dims shape{};
    shape.nbDims = static_cast<int>(dims.size());
    std::copy(dims.begin(), dims.end(), shape.d);
#if NV_TENSORRT_MAJOR >= 10
    return context_->context_->setInputShape(tensor_name(ibinding), shape);
#else
    return this->context_->context_->setBindingDimensions(ibinding, shape);
#endif
  }

  virtual int numel(const std::string &name) override {
    return numel(index(name));
  }

  virtual int numel(int ibinding) override {
#if NV_TENSORRT_MAJOR >= 10
    auto dim = context_->context_->getTensorShape(tensor_name(ibinding));
#else
    auto dim = this->context_->context_->getBindingDimensions(ibinding);
#endif
    return std::accumulate(dim.d, dim.d + dim.nbDims, 1,
                           std::multiplies<int>());
  }

  virtual DType dtype(const std::string &name) override {
    return dtype(index(name));
  }

  virtual DType dtype(int ibinding) override {
#if NV_TENSORRT_MAJOR >= 10
    return (DType)context_->engine_->getTensorDataType(tensor_name(ibinding));
#else
    return (DType)this->context_->engine_->getBindingDataType(ibinding);
#endif
  }

  virtual bool has_dynamic_dim() override {
    // check if any input or output bindings have dynamic shapes
    // code from ChatGPT
    int numBindings = num_bindings();
    for (int i = 0; i < numBindings; ++i) {
#if NV_TENSORRT_MAJOR >= 10
      nvinfer1::Dims dims =
          this->context_->engine_->getTensorShape(tensor_name(i));
#else
      nvinfer1::Dims dims = this->context_->engine_->getBindingDimensions(i);
#endif
      for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] == -1)
          return true;
      }
    }
    return false;
  }

  virtual void print(const char *name) override {
    printf("------------------------------------------------------\n");
    printf("%s 🌱 is %s model\n", name,
           has_dynamic_dim() ? "Dynamic Shape" : "Static Shape");

    int num_input = 0;
    int num_output = 0;
    auto engine = this->context_->engine_;
#if NV_TENSORRT_MAJOR >= 10
    for (int i = 0; i < num_bindings(); ++i) {
      if (is_input(i))
        num_input++;
      else
        num_output++;
    }
    printf("Inputs: %d\n", num_input);
    for (int i = 0; i < num_bindings(); ++i) {
      if (is_input(i)) {
        const char *tensor = tensor_name(i);
        auto dim = engine->getTensorShape(tensor);
        auto dtype = engine->getTensorDataType(tensor);
        printf("\t%d.%s : {%s} [%s]\n", i, tensor, format_shape(dim).c_str(),
               data_type_string(dtype));
      }
    }
    printf("Outputs: %d\n", num_output);
    for (int i = 0; i < num_bindings(); ++i) {
      if (!is_input(i)) {
        const char *tensor = tensor_name(i);
        auto dim = engine->getTensorShape(tensor);
        auto dtype = engine->getTensorDataType(tensor);
        printf("\t%d.%s : {%s} [%s]\n", i, tensor, format_shape(dim).c_str(),
               data_type_string(dtype));
      }
    }
#else
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      if (engine->bindingIsInput(i))
        num_input++;
      else
        num_output++;
    }

    printf("Inputs: %d\n", num_input);
    for (int i = 0; i < num_input; ++i) {
      auto name = engine->getBindingName(i);
      auto dim = engine->getBindingDimensions(i);
      auto dtype = engine->getBindingDataType(i);
      printf("\t%d.%s : {%s} [%s]\n", i, name, format_shape(dim).c_str(),
             data_type_string(dtype));
    }

    printf("Outputs: %d\n", num_output);
    for (int i = 0; i < num_output; ++i) {
      auto name = engine->getBindingName(i + num_input);
      auto dim = engine->getBindingDimensions(i + num_input);
      auto dtype = engine->getBindingDataType(i + num_input);
      printf("\t%d.%s : {%s} [%s]\n", i, name, format_shape(dim).c_str(),
             data_type_string(dtype));
    }
#endif
    printf("------------------------------------------------------\n");
  }

#if NV_TENSORRT_MAJOR >= 10
private:
  std::vector<std::string> tensor_names_;
#endif
};

std::shared_ptr<Engine> load(const std::string &file) {
  std::shared_ptr<EngineImplement> impl(new EngineImplement());
  if (!impl->load(file))
    impl.reset();
  return impl;
}

}; // namespace TensorRT
