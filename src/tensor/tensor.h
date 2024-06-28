#pragma once

#include <napi.h>

#include <torch/torch.h>

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

  Napi::Value isComplex(const Napi::CallbackInfo &);
  Napi::Value isConj(const Napi::CallbackInfo &);
  Napi::Value isFloatingPoint(const Napi::CallbackInfo &);
  Napi::Value isNonzero(const Napi::CallbackInfo &);

  torch::Tensor tensor();

private:
  torch::Tensor tensor_;

  Napi::Value abs(const Napi::CallbackInfo &);
  Napi::Value acos(const Napi::CallbackInfo &);
  Napi::Value arccos(const Napi::CallbackInfo &);
  Napi::Value adjoint(const Napi::CallbackInfo &);
  Napi::Value angle(const Napi::CallbackInfo &);
  Napi::Value asin(const Napi::CallbackInfo &);
  Napi::Value arcsin(const Napi::CallbackInfo &);
  Napi::Value atan(const Napi::CallbackInfo &);
  Napi::Value arctan(const Napi::CallbackInfo &);
  Napi::Value ceil(const Napi::CallbackInfo &);
  Napi::Value clone(const Napi::CallbackInfo &);
  Napi::Value conj(const Napi::CallbackInfo &);
  Napi::Value conj_physical(const Napi::CallbackInfo &);
  Napi::Value resolve_conj(const Napi::CallbackInfo &);
  Napi::Value resolve_neg(const Napi::CallbackInfo &);
  Napi::Value cos(const Napi::CallbackInfo &);
  Napi::Value corrcoef(const Napi::CallbackInfo &);
  Napi::Value cosh(const Napi::CallbackInfo &);
  Napi::Value acosh(const Napi::CallbackInfo &);
  Napi::Value arccosh(const Napi::CallbackInfo &);
  Napi::Value det(const Napi::CallbackInfo &);
  Napi::Value digamma(const Napi::CallbackInfo &);
  Napi::Value erf(const Napi::CallbackInfo &);
  Napi::Value erfc(const Napi::CallbackInfo &);
  Napi::Value erfinv(const Napi::CallbackInfo &);
  Napi::Value exp(const Napi::CallbackInfo &);
  Napi::Value expm1(const Napi::CallbackInfo &);
  Napi::Value fix(const Napi::CallbackInfo &);
  Napi::Value fliplr(const Napi::CallbackInfo &);
  Napi::Value flipud(const Napi::CallbackInfo &);
  Napi::Value floor(const Napi::CallbackInfo &);
  Napi::Value frac(const Napi::CallbackInfo &);
  Napi::Value i0(const Napi::CallbackInfo &);
  Napi::Value indices(const Napi::CallbackInfo &);
  Napi::Value int_repr(const Napi::CallbackInfo &);
  Napi::Value inverse(const Napi::CallbackInfo &);
  Napi::Value isfinite(const Napi::CallbackInfo &);
  Napi::Value isinf(const Napi::CallbackInfo &);
  Napi::Value isposinf(const Napi::CallbackInfo &);
  Napi::Value isneginf(const Napi::CallbackInfo &);
  Napi::Value isnan(const Napi::CallbackInfo &);
  Napi::Value lgamma(const Napi::CallbackInfo &);
  Napi::Value log(const Napi::CallbackInfo &);
  Napi::Value logdet(const Napi::CallbackInfo &);
  Napi::Value log10(const Napi::CallbackInfo &);
  Napi::Value log1p(const Napi::CallbackInfo &);
  Napi::Value log2(const Napi::CallbackInfo &);
  Napi::Value logit(const Napi::CallbackInfo &);
  Napi::Value neg(const Napi::CallbackInfo &);
  Napi::Value negative(const Napi::CallbackInfo &);
  Napi::Value pin_memory(const Napi::CallbackInfo &);
  Napi::Value pinverse(const Napi::CallbackInfo &);
  Napi::Value positive(const Napi::CallbackInfo &);
  Napi::Value rad2deg(const Napi::CallbackInfo &);
  Napi::Value ravel(const Napi::CallbackInfo &);
  Napi::Value reciprocal(const Napi::CallbackInfo &);
  Napi::Value rsqrt(const Napi::CallbackInfo &);
  Napi::Value sigmoid(const Napi::CallbackInfo &);
  Napi::Value sign(const Napi::CallbackInfo &);
  Napi::Value signbit(const Napi::CallbackInfo &);
  Napi::Value sgn(const Napi::CallbackInfo &);
  Napi::Value sin(const Napi::CallbackInfo &);
  Napi::Value sinc(const Napi::CallbackInfo &);
  Napi::Value sinh(const Napi::CallbackInfo &);
  Napi::Value asinh(const Napi::CallbackInfo &);
  Napi::Value arcsinh(const Napi::CallbackInfo &);
  Napi::Value sqrt(const Napi::CallbackInfo &);
  Napi::Value square(const Napi::CallbackInfo &);
  Napi::Value t(const Napi::CallbackInfo &);
  Napi::Value tan(const Napi::CallbackInfo &);
  Napi::Value tanh(const Napi::CallbackInfo &);
  Napi::Value arctanh(const Napi::CallbackInfo &);
  Napi::Value toString(const Napi::CallbackInfo &);
  Napi::Value trace(const Napi::CallbackInfo &);
  Napi::Value trunc(const Napi::CallbackInfo &);
  Napi::Value values(const Napi::CallbackInfo &);
};

} // namespace tensor

} // namespace jtorch
