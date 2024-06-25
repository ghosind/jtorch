#pragma once

#include <napi.h>

#include <torch/torch.h>

namespace jtorch {

class Tensor : public Napi::ObjectWrap<Tensor> {
public:
  static Napi::Function GetClass(Napi::Env);
  static Napi::Object Init(Napi::Env, Napi::Object exports);

  Tensor(const Napi::CallbackInfo &);

private:
  torch::Tensor tensor;

  Napi::Value toString(const Napi::CallbackInfo &);
};

}
