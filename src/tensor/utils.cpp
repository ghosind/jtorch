#include "tensor.h"

#include <napi.h>

namespace jtorch {

namespace tensor {

bool validateTensorParam(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() == 0) {
    Napi::TypeError::New(env, "missing 1 required positional argument")
      .ThrowAsJavaScriptException();
    return false;
  } else if (!Tensor::IsTensor(env, info[0])) {
    Napi::TypeError::New(env, "argument must be Tensor")
      .ThrowAsJavaScriptException();
    return false;
  }

  return true;
}

} // namespace tensor

} // namespace jtorch
