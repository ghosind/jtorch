#pragma once

#include <napi.h>

#include <torch/torch.h>

#ifndef TENSOR_METHOD_TENSOR
#define TENSOR_METHOD_TENSOR(FNAME) \
inline Napi::Value FNAME(const Napi::CallbackInfo &info) {\
  torch::Tensor t = tensor_.FNAME(); \
  return Tensor::New(info, t); \
}
#endif // TENSOR_METHOD_TENSOR

#ifndef TENSOR_METHOD_BOOL
#define TENSOR_METHOD_BOOL(FNAME) \
inline Napi::Value FNAME(const Napi::CallbackInfo &info) {\
  Napi::Env env = info.Env(); \
  bool ret = tensor_.FNAME(); \
  return Napi::Boolean::New(env, ret); \
}
#endif // TENSOR_METHOD_BOOL

namespace jtorch {

namespace tensor {

class Tensor : public Napi::ObjectWrap<Tensor> {
public:
  static Napi::Function GetClass(Napi::Env);
  static Napi::Object Init(Napi::Env, Napi::Object);
  static Napi::Object New(const Napi::CallbackInfo &, const torch::Tensor &);
  static Napi::Object New(Napi::Env, const torch::Tensor &);

  static bool IsTensor(Napi::Env, Napi::Value);
  static Tensor *AsTensor(Napi::Object);

  Tensor(const Napi::CallbackInfo &);

  TENSOR_METHOD_BOOL(is_complex)
  TENSOR_METHOD_BOOL(is_conj)
  TENSOR_METHOD_BOOL(is_floating_point)
  TENSOR_METHOD_BOOL(is_nonzero)

  torch::Tensor tensor();

private:
  torch::Tensor tensor_;

  Napi::Value toString(const Napi::CallbackInfo &);

  TENSOR_METHOD_TENSOR(abs)
  TENSOR_METHOD_TENSOR(acos)
  TENSOR_METHOD_TENSOR(arccos)
  TENSOR_METHOD_TENSOR(adjoint)
  TENSOR_METHOD_TENSOR(angle)
  TENSOR_METHOD_TENSOR(asin)
  TENSOR_METHOD_TENSOR(arcsin)
  TENSOR_METHOD_TENSOR(atan)
  TENSOR_METHOD_TENSOR(arctan)
  TENSOR_METHOD_TENSOR(ceil)
  TENSOR_METHOD_TENSOR(clone)
  TENSOR_METHOD_TENSOR(conj)
  TENSOR_METHOD_TENSOR(conj_physical)
  TENSOR_METHOD_TENSOR(resolve_conj)
  TENSOR_METHOD_TENSOR(resolve_neg)
  TENSOR_METHOD_TENSOR(cos)
  TENSOR_METHOD_TENSOR(corrcoef)
  TENSOR_METHOD_TENSOR(cosh)
  TENSOR_METHOD_TENSOR(acosh)
  TENSOR_METHOD_TENSOR(arccosh)
  TENSOR_METHOD_TENSOR(det)
  TENSOR_METHOD_TENSOR(digamma)
  TENSOR_METHOD_TENSOR(erf)
  TENSOR_METHOD_TENSOR(erfc)
  TENSOR_METHOD_TENSOR(erfinv)
  TENSOR_METHOD_TENSOR(exp)
  TENSOR_METHOD_TENSOR(expm1)
  TENSOR_METHOD_TENSOR(fix)
  TENSOR_METHOD_TENSOR(fliplr)
  TENSOR_METHOD_TENSOR(flipud)
  TENSOR_METHOD_TENSOR(floor)
  TENSOR_METHOD_TENSOR(frac)
  TENSOR_METHOD_TENSOR(i0)
  TENSOR_METHOD_TENSOR(indices)
  TENSOR_METHOD_TENSOR(int_repr)
  TENSOR_METHOD_TENSOR(inverse)
  TENSOR_METHOD_TENSOR(isfinite)
  TENSOR_METHOD_TENSOR(isinf)
  TENSOR_METHOD_TENSOR(isposinf)
  TENSOR_METHOD_TENSOR(isneginf)
  TENSOR_METHOD_TENSOR(isnan)
  TENSOR_METHOD_TENSOR(lgamma)
  TENSOR_METHOD_TENSOR(log)
  TENSOR_METHOD_TENSOR(logdet)
  TENSOR_METHOD_TENSOR(log10)
  TENSOR_METHOD_TENSOR(log1p)
  TENSOR_METHOD_TENSOR(log2)
  TENSOR_METHOD_TENSOR(logit)
  TENSOR_METHOD_TENSOR(neg)
  TENSOR_METHOD_TENSOR(negative)
  TENSOR_METHOD_TENSOR(pin_memory)
  TENSOR_METHOD_TENSOR(pinverse)
  TENSOR_METHOD_TENSOR(positive)
  TENSOR_METHOD_TENSOR(rad2deg)
  TENSOR_METHOD_TENSOR(ravel)
  TENSOR_METHOD_TENSOR(reciprocal)
  TENSOR_METHOD_TENSOR(rsqrt)
  TENSOR_METHOD_TENSOR(sigmoid)
  TENSOR_METHOD_TENSOR(sign)
  TENSOR_METHOD_TENSOR(signbit)
  TENSOR_METHOD_TENSOR(sgn)
  TENSOR_METHOD_TENSOR(sin)
  TENSOR_METHOD_TENSOR(sinc)
  TENSOR_METHOD_TENSOR(sinh)
  TENSOR_METHOD_TENSOR(asinh)
  TENSOR_METHOD_TENSOR(arcsinh)
  TENSOR_METHOD_TENSOR(sqrt)
  TENSOR_METHOD_TENSOR(square)
  TENSOR_METHOD_TENSOR(t)
  TENSOR_METHOD_TENSOR(tan)
  TENSOR_METHOD_TENSOR(tanh)
  TENSOR_METHOD_TENSOR(arctanh)
  TENSOR_METHOD_TENSOR(trace)
  TENSOR_METHOD_TENSOR(trunc)
  TENSOR_METHOD_TENSOR(values)
};

} // namespace tensor

} // namespace jtorch
