#include "tensor.h"

#include <torch/torch.h>

namespace jtorch {

Tensor::Tensor(const Napi::CallbackInfo &info) : ObjectWrap(info) {
  tensor = torch::empty(0);
}

Napi::Value Tensor::toString(const Napi::CallbackInfo &info) {
  Napi::Value ret;
  try {
    ret = Napi::String::New(info.Env(), tensor.toString());
  } catch (const std::exception &e) {
    Napi::Error::New(info.Env(), e.what()).ThrowAsJavaScriptException();
  }
  return ret;
}

Napi::Function Tensor::GetClass(Napi::Env env) {
  return DefineClass(env, "Tensor", {
    InstanceMethod("toString", &Tensor::toString),
  });
}

Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports) {
  Napi::String name = Napi::String::New(env, "Tensor");
  exports.Set(name, Tensor::GetClass(env));

  return exports;
}

}
