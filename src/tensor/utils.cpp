#include "tensor.h"
#include "../node_api.h"

#include <napi.h>

namespace jtorch {

namespace tensor {

bool validateTensorParam(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() == 0) {
    THROW_TYPE_ERROR(env, "missing 1 required positional argument");
    return false;
  } else if (!Tensor::IsTensor(env, info[0])) {
    THROW_TYPE_ERROR(env, "argument must be Tensor");
    return false;
  }

  return true;
}

} // namespace tensor

} // namespace jtorch
