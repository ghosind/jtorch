#include "creation.h"
#include "tensor.h"
#include "utils.h"
#include "../node_api.h"

#include <napi.h>

namespace jtorch {

namespace tensor {

Napi::Value isComplex(const Napi::CallbackInfo &info) {
  if (!validateTensorParam(info)) {
    Napi::Value ret;
    return ret;
  }

  return TENSOR_VALUE(info[0])->is_complex(info);
}

Napi::Value isConj(const Napi::CallbackInfo &info) {
  if (!validateTensorParam(info)) {
    Napi::Value ret;
    return ret;
  }

  return TENSOR_VALUE(info[0])->is_conj(info);
}

Napi::Value isFloatingPoint(const Napi::CallbackInfo &info) {
  if (!validateTensorParam(info)) {
    Napi::Value ret;
    return ret;
  }

  return TENSOR_VALUE(info[0])->is_floating_point(info);
}

Napi::Value isNonzero(const Napi::CallbackInfo &info) {
  if (!validateTensorParam(info)) {
    Napi::Value ret;
    return ret;
  }

  return TENSOR_VALUE(info[0])->is_nonzero(info);
}

Napi::Value isTensor(const Napi::CallbackInfo &info) {
  Napi::Value ret;
  Napi::Env env = info.Env();

  if (info.Length() == 0) {
    THROW_TYPE_ERROR(env, "missing 1 required positional argument");
    return ret;
  }

  bool res = Tensor::IsTensor(env, info[0]);
  return Napi::Boolean::New(env, res);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Tensor::Init(env, exports);

  exports.Set(Napi::String::New(env, "arange"), Napi::Function::New(env, arange));

  exports.Set(Napi::String::New(env, "isComplex"), Napi::Function::New(env, isComplex));
  exports.Set(Napi::String::New(env, "isConj"), Napi::Function::New(env, isConj));
  exports.Set(Napi::String::New(env, "isFloatingPoint"), Napi::Function::New(env, isFloatingPoint));
  exports.Set(Napi::String::New(env, "isNonzero"), Napi::Function::New(env, isNonzero));
  exports.Set(Napi::String::New(env, "isTensor"), Napi::Function::New(env, isTensor));

  return exports;
}

} // tensor

} // jtorch
