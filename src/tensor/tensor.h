#pragma once

#include <napi.h>

#include <torch/torch.h>

namespace jtorch {

class Tensor : public Napi::ObjectWrap<Tensor> {
public:
  static Napi::Function GetClass(Napi::Env);
  static Napi::Object Init(Napi::Env, Napi::Object);
  static Napi::Object New(const Napi::CallbackInfo &, const torch::Tensor &);
  static Napi::Object New(Napi::Env, const torch::Tensor &);

  Tensor(const Napi::CallbackInfo &);

  torch::Tensor tensor();

private:
  torch::Tensor tensor_;

  Napi::Value t(const Napi::CallbackInfo &);
  Napi::Value toString(const Napi::CallbackInfo &);
};

}
