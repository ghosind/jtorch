#include "tensor.h"
#include "utils.h"
#include "../node_api.h"

#include <napi.h>
#include <torch/torch.h>

namespace jtorch {

namespace tensor {

Napi::Value arange(const Napi::CallbackInfo &info) {
  Napi::Value ret;
  Napi::Env env = info.Env();
  torch::Tensor tensor;

  int len = info.Length();
  if (len == 0 || len > 4) {
    THROW_TYPE_ERROR(env, "Invalid argument");
    return ret;
  }

  torch::TensorOptions options = parseTensorOptions(info[len-1]);
  int i = 0;

  for (; i < len; i++) {
    if (!info[i].IsNumber()) {
      break;
    }
  }
  switch (i) {
  case 1:
    tensor = torch::arange(INT64_VALUE(info[0]), options);
    break;
  case 2:
    tensor = torch::arange(INT64_VALUE(info[0]), INT64_VALUE(info[1]), options);
    break;
  case 3:
    tensor = torch::arange(INT64_VALUE(info[0]), INT64_VALUE(info[1]), INT64_VALUE(info[2]), options);
    break;
  default:
    THROW_TYPE_ERROR(env, "Invalid argument");
    return ret;
  }

  return Tensor::New(info, tensor);
}

} // namespace tensor

} // namespace jtorch
