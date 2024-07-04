#pragma once

#include <napi.h>

#define TENSOR_VALUE(val) Tensor::AsTensor(val.ToObject())

namespace jtorch {

namespace tensor {

bool validateTensorParam(const Napi::CallbackInfo &);

} // namespace tensor

} // namespace jtorch
