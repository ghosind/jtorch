#pragma once

#include <napi.h>
#include <torch/torch.h>

#define TENSOR_VALUE(val) Tensor::AsTensor(val.ToObject())

namespace jtorch {

namespace tensor {

torch::TensorOptions parseTensorOptions(Napi::Value);

bool validateTensorParam(const Napi::CallbackInfo &);

} // namespace tensor

} // namespace jtorch
