#include "tensor.h"

#include <torch/torch.h>

namespace jtorch {

Napi::Function Tensor::GetClass(Napi::Env env) {
  return DefineClass(env, "Tensor", {
    InstanceMethod("t", &Tensor::t),
    InstanceMethod("toString", &Tensor::toString),
  });
}

Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports) {
  Napi::String name = Napi::String::New(env, "Tensor");
  Napi::Function func = Tensor::GetClass(env);

  Napi::FunctionReference *constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);
  env.SetInstanceData(constructor);

  exports.Set(name, func);

  return exports;
}

Napi::Object Tensor::New(const Napi::CallbackInfo &info, const torch::Tensor &tensor) {
  Napi::Env env = info.Env();
  return Tensor::New(env, tensor);
}

Napi::Object Tensor::New(Napi::Env env, const torch::Tensor &tensor) {
  Napi::Object ret;

  try {
    Napi::EscapableHandleScope scope(env);
    Napi::Object obj = env.GetInstanceData<Napi::FunctionReference>()->New({});
    Napi::ObjectWrap<Tensor>::Unwrap(obj)->tensor_ = tensor;
    ret = scope.Escape(obj).ToObject();
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }

  return ret;
}

Tensor::Tensor(const Napi::CallbackInfo &info) : ObjectWrap(info) {
  if (info.Length() == 0) {
    tensor_ = torch::empty(0);
    return;
  }

  // TODO:
  // 1st parameter: initial data
  // 2nd parameter: opts

  Napi::Env env = info.Env();
  Napi::TypeError::New(env, "Unsupported init").ThrowAsJavaScriptException();
}

torch::Tensor Tensor::tensor() {
  return tensor_;
}

Napi::Value Tensor::t(const Napi::CallbackInfo &info) {
  torch::Tensor t = tensor_.t();
  return Tensor::New(info, t);
}

Napi::Value Tensor::toString(const Napi::CallbackInfo &info) {
  // TODO: more clear output

  std::ostringstream stream;
  stream << tensor_;
  std::string str = stream.str();
  return Napi::String::New(info.Env(), str);
}

} // namespace jtorch
