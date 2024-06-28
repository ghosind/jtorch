#include "tensor.h"
#include "utils.h"

#include <torch/torch.h>

namespace jtorch {

namespace tensor {

Napi::Function Tensor::GetClass(Napi::Env env) {
  return DefineClass(env, "Tensor", {
    InstanceAccessor<&Tensor::t>("T"),
    InstanceMethod("abs", &Tensor::abs),
    InstanceMethod("absolute", &Tensor::abs),
    InstanceMethod("acos", &Tensor::acos),
    InstanceMethod("arccos", &Tensor::arccos),
    InstanceMethod("adjoint", &Tensor::adjoint),
    InstanceMethod("angle", &Tensor::angle),
    InstanceMethod("asin", &Tensor::asin),
    InstanceMethod("arcsin", &Tensor::arcsin),
    InstanceMethod("atan", &Tensor::atan),
    InstanceMethod("arctan", &Tensor::arctan),
    InstanceMethod("ceil", &Tensor::ceil),
    InstanceMethod("clone", &Tensor::clone),
    InstanceMethod("conj", &Tensor::conj),
    InstanceMethod("conj_physical", &Tensor::conj_physical),
    InstanceMethod("resolve_conj", &Tensor::resolve_conj),
    InstanceMethod("resolve_neg", &Tensor::resolve_neg),
    InstanceMethod("cos", &Tensor::cos),
    InstanceMethod("corrcoef", &Tensor::corrcoef),
    InstanceMethod("cosh", &Tensor::cosh),
    InstanceMethod("acosh", &Tensor::acosh),
    InstanceMethod("arccosh", &Tensor::arccosh),
    InstanceMethod("det", &Tensor::det),
    InstanceMethod("digamma", &Tensor::digamma),
    InstanceMethod("erf", &Tensor::erf),
    InstanceMethod("erfc", &Tensor::erfc),
    InstanceMethod("erfinv", &Tensor::erfinv),
    InstanceMethod("exp", &Tensor::exp),
    InstanceMethod("expm1", &Tensor::expm1),
    InstanceMethod("fix", &Tensor::fix),
    InstanceMethod("fliplr", &Tensor::fliplr),
    InstanceMethod("flipud", &Tensor::flipud),
    InstanceMethod("floor", &Tensor::floor),
    InstanceMethod("frac", &Tensor::frac),
    InstanceMethod("i0", &Tensor::i0),
    InstanceMethod("indices", &Tensor::indices),
    InstanceMethod("int_repr", &Tensor::int_repr),
    InstanceMethod("inverse", &Tensor::inverse),
    InstanceMethod("isfinite", &Tensor::isfinite),
    InstanceMethod("isinf", &Tensor::isinf),
    InstanceMethod("isposinf", &Tensor::isposinf),
    InstanceMethod("isneginf", &Tensor::isneginf),
    InstanceMethod("isnan", &Tensor::isnan),
    InstanceMethod("isComplex", &Tensor::isComplex),
    InstanceMethod("isConj", &Tensor::isConj),
    InstanceMethod("isFloatingPoint", &Tensor::isFloatingPoint),
    InstanceMethod("isNonzero", &Tensor::isNonzero),
    InstanceMethod("lgamma", &Tensor::lgamma),
    InstanceMethod("log", &Tensor::log),
    InstanceMethod("logdet", &Tensor::logdet),
    InstanceMethod("log10", &Tensor::log10),
    InstanceMethod("log1p", &Tensor::log1p),
    InstanceMethod("log2", &Tensor::log2),
    InstanceMethod("logit", &Tensor::logit),
    InstanceMethod("neg", &Tensor::neg),
    InstanceMethod("negative", &Tensor::negative),
    InstanceMethod("pin_memory", &Tensor::pin_memory),
    InstanceMethod("pinverse", &Tensor::pinverse),
    InstanceMethod("positive", &Tensor::positive),
    InstanceMethod("rad2deg", &Tensor::rad2deg),
    InstanceMethod("ravel", &Tensor::ravel),
    InstanceMethod("reciprocal", &Tensor::reciprocal),
    InstanceMethod("rsqrt", &Tensor::rsqrt),
    InstanceMethod("sigmoid", &Tensor::sigmoid),
    InstanceMethod("sign", &Tensor::sign),
    InstanceMethod("signbit", &Tensor::signbit),
    InstanceMethod("sgn", &Tensor::sgn),
    InstanceMethod("sin", &Tensor::sin),
    InstanceMethod("sinc", &Tensor::sinc),
    InstanceMethod("sinh", &Tensor::sinh),
    InstanceMethod("asinh", &Tensor::asinh),
    InstanceMethod("arcsinh", &Tensor::arcsinh),
    InstanceMethod("sqrt", &Tensor::sqrt),
    InstanceMethod("square", &Tensor::square),
    InstanceMethod("t", &Tensor::t),
    InstanceMethod("tan", &Tensor::tan),
    InstanceMethod("tanh", &Tensor::tanh),
    InstanceMethod("arctanh", &Tensor::arctanh),
    InstanceMethod("toString", &Tensor::toString),
    InstanceMethod("trace", &Tensor::trace),
    InstanceMethod("trunc", &Tensor::trunc),
    InstanceMethod("values", &Tensor::values),
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

bool Tensor::IsTensor(Napi::Env env, Napi::Value val) {
  if (!val.IsObject()) {
    return false;
  }

  Napi::FunctionReference *constructor = env.GetInstanceData<Napi::FunctionReference>();

  return val.ToObject().InstanceOf(constructor->Value());
}

Tensor *Tensor::AsTensor(Napi::Object obj) {
  return Napi::ObjectWrap<Tensor>::Unwrap(obj);
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
  Napi::Value data = info[0];
  if (Tensor::IsTensor(env, data)) {
    tensor_ = Tensor::AsTensor(data.ToObject())->tensor().clone();
    return;
  }

  Napi::TypeError::New(env, "Unsupported init").ThrowAsJavaScriptException();
}

torch::Tensor Tensor::tensor() {
  return tensor_;
}

Napi::Value Tensor::abs(const Napi::CallbackInfo &info) {
  torch::Tensor absTensor = tensor_.abs();
  return Tensor::New(info, absTensor);
}

Napi::Value Tensor::acos(const Napi::CallbackInfo &info) {
  torch::Tensor acosTensor = tensor_.acos();
  return Tensor::New(info, acosTensor);
}

Napi::Value Tensor::arccos(const Napi::CallbackInfo &info) {
  torch::Tensor arccosTensor = tensor_.arccos();
  return Tensor::New(info, arccosTensor);
}

Napi::Value Tensor::adjoint(const Napi::CallbackInfo &info) {
  torch::Tensor adjointTensor = tensor_.adjoint();
  return Tensor::New(info, adjointTensor);
}

Napi::Value Tensor::angle(const Napi::CallbackInfo &info) {
  torch::Tensor angleTensor = tensor_.angle();
  return Tensor::New(info, angleTensor);
}

Napi::Value Tensor::asin(const Napi::CallbackInfo &info) {
  torch::Tensor asinTensor = tensor_.asin();
  return Tensor::New(info, asinTensor);
}

Napi::Value Tensor::arcsin(const Napi::CallbackInfo &info) {
  torch::Tensor arcsinTensor = tensor_.arcsin();
  return Tensor::New(info, arcsinTensor);
}

Napi::Value Tensor::atan(const Napi::CallbackInfo &info) {
  torch::Tensor atanTensor = tensor_.atan();
  return Tensor::New(info, atanTensor);
}

Napi::Value Tensor::arctan(const Napi::CallbackInfo &info) {
  torch::Tensor arctanTensor = tensor_.arctan();
  return Tensor::New(info, arctanTensor);
}

Napi::Value Tensor::ceil(const Napi::CallbackInfo &info) {
  torch::Tensor ceilTensor = tensor_.ceil();
  return Tensor::New(info, ceilTensor);
}

Napi::Value Tensor::clone(const Napi::CallbackInfo &info) {
  torch::Tensor clonedTensor = tensor_.clone();
  return Tensor::New(info, clonedTensor);
}

Napi::Value Tensor::conj(const Napi::CallbackInfo &info) {
  torch::Tensor conjTensor = tensor_.clone();
  return Tensor::New(info, conjTensor);
}

Napi::Value Tensor::conj_physical(const Napi::CallbackInfo &info) {
  torch::Tensor conjTensor = tensor_.conj_physical();
  return Tensor::New(info, conjTensor);
}

Napi::Value Tensor::resolve_conj(const Napi::CallbackInfo &info) {
  torch::Tensor conjTensor = tensor_.resolve_conj();
  return Tensor::New(info, conjTensor);
}

Napi::Value Tensor::resolve_neg(const Napi::CallbackInfo &info) {
  torch::Tensor negTensor = tensor_.resolve_neg();
  return Tensor::New(info, negTensor);
}

Napi::Value Tensor::cos(const Napi::CallbackInfo &info) {
  torch::Tensor cosTensor = tensor_.cos();
  return Tensor::New(info, cosTensor);
}

Napi::Value Tensor::corrcoef(const Napi::CallbackInfo &info) {
  torch::Tensor corrcoefTensor = tensor_.corrcoef();
  return Tensor::New(info, corrcoefTensor);
}

Napi::Value Tensor::cosh(const Napi::CallbackInfo &info) {
  torch::Tensor coshTensor = tensor_.cosh();
  return Tensor::New(info, coshTensor);
}

Napi::Value Tensor::acosh(const Napi::CallbackInfo &info) {
  torch::Tensor acoshTensor = tensor_.acosh();
  return Tensor::New(info, acoshTensor);
}

Napi::Value Tensor::arccosh(const Napi::CallbackInfo &info) {
  torch::Tensor arccoshTensor = tensor_.arccosh();
  return Tensor::New(info, arccoshTensor);
}

Napi::Value Tensor::det(const Napi::CallbackInfo &info) {
  torch::Tensor detTensor = tensor_.det();
  return Tensor::New(info, detTensor);
}

Napi::Value Tensor::digamma(const Napi::CallbackInfo &info) {
  torch::Tensor digammaTensor = tensor_.digamma();
  return Tensor::New(info, digammaTensor);
}

Napi::Value Tensor::erf(const Napi::CallbackInfo &info) {
  torch::Tensor erfTensor = tensor_.erf();
  return Tensor::New(info, erfTensor);
}

Napi::Value Tensor::erfc(const Napi::CallbackInfo &info) {
  torch::Tensor erfcTensor = tensor_.erfc();
  return Tensor::New(info, erfcTensor);
}

Napi::Value Tensor::erfinv(const Napi::CallbackInfo &info) {
  torch::Tensor erfinvTensor = tensor_.erfinv();
  return Tensor::New(info, erfinvTensor);
}

Napi::Value Tensor::exp(const Napi::CallbackInfo &info) {
  torch::Tensor expTensor = tensor_.exp();
  return Tensor::New(info, expTensor);
}

Napi::Value Tensor::expm1(const Napi::CallbackInfo &info) {
  torch::Tensor expm1Tensor = tensor_.expm1();
  return Tensor::New(info, expm1Tensor);
}

Napi::Value Tensor::fix(const Napi::CallbackInfo &info) {
  torch::Tensor fixTensor = tensor_.fix();
  return Tensor::New(info, fixTensor);
}

Napi::Value Tensor::fliplr(const Napi::CallbackInfo &info) {
  torch::Tensor fliplrTensor = tensor_.fliplr();
  return Tensor::New(info, fliplrTensor);
}

Napi::Value Tensor::flipud(const Napi::CallbackInfo &info) {
  torch::Tensor flipudTensor = tensor_.flipud();
  return Tensor::New(info, flipudTensor);
}

Napi::Value Tensor::floor(const Napi::CallbackInfo &info) {
  torch::Tensor floorTensor = tensor_.floor();
  return Tensor::New(info, floorTensor);
}

Napi::Value Tensor::frac(const Napi::CallbackInfo &info) {
  torch::Tensor fracTensor = tensor_.frac();
  return Tensor::New(info, fracTensor);
}

Napi::Value Tensor::i0(const Napi::CallbackInfo &info) {
  torch::Tensor i0Tensor = tensor_.i0();
  return Tensor::New(info, i0Tensor);
}

Napi::Value Tensor::indices(const Napi::CallbackInfo &info) {
  torch::Tensor indicesTensor = tensor_.indices();
  return Tensor::New(info, indicesTensor);
}

Napi::Value Tensor::int_repr(const Napi::CallbackInfo &info) {
  torch::Tensor int_reprTensor = tensor_.int_repr();
  return Tensor::New(info, int_reprTensor);
}

Napi::Value Tensor::inverse(const Napi::CallbackInfo &info) {
  torch::Tensor inverseTensor = tensor_.inverse();
  return Tensor::New(info, inverseTensor);
}

Napi::Value Tensor::isfinite(const Napi::CallbackInfo &info) {
  torch::Tensor isfiniteTensor = tensor_.isfinite();
  return Tensor::New(info, isfiniteTensor);
}

Napi::Value Tensor::isinf(const Napi::CallbackInfo &info) {
  torch::Tensor isinfTensor = tensor_.isinf();
  return Tensor::New(info, isinfTensor);
}

Napi::Value Tensor::isposinf(const Napi::CallbackInfo &info) {
  torch::Tensor isposinfTensor = tensor_.isposinf();
  return Tensor::New(info, isposinfTensor);
}

Napi::Value Tensor::isneginf(const Napi::CallbackInfo &info) {
  torch::Tensor isneginfTensor = tensor_.isneginf();
  return Tensor::New(info, isneginfTensor);
}

Napi::Value Tensor::isnan(const Napi::CallbackInfo &info) {
  torch::Tensor isnanTensor = tensor_.isnan();
  return Tensor::New(info, isnanTensor);
}

Napi::Value Tensor::isComplex(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  bool ret = tensor_.is_complex();
  return Napi::Boolean::New(env, ret);
}

Napi::Value Tensor::isConj(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  bool ret = tensor_.is_conj();
  return Napi::Boolean::New(env, ret);
}

Napi::Value Tensor::isFloatingPoint(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  bool ret = tensor_.is_floating_point();
  return Napi::Boolean::New(env, ret);
}

Napi::Value Tensor::isNonzero(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  bool ret = tensor_.is_nonzero();
  return Napi::Boolean::New(env, ret);
}

Napi::Value Tensor::lgamma(const Napi::CallbackInfo &info) {
  torch::Tensor lgammaTensor = tensor_.lgamma();
  return Tensor::New(info, lgammaTensor);
}

Napi::Value Tensor::log(const Napi::CallbackInfo &info) {
  torch::Tensor logTensor = tensor_.log();
  return Tensor::New(info, logTensor);
}

Napi::Value Tensor::logdet(const Napi::CallbackInfo &info) {
  torch::Tensor logdetTensor = tensor_.logdet();
  return Tensor::New(info, logdetTensor);
}

Napi::Value Tensor::log10(const Napi::CallbackInfo &info) {
  torch::Tensor log10Tensor = tensor_.log10();
  return Tensor::New(info, log10Tensor);
}

Napi::Value Tensor::log1p(const Napi::CallbackInfo &info) {
  torch::Tensor log1pTensor = tensor_.log1p();
  return Tensor::New(info, log1pTensor);
}

Napi::Value Tensor::log2(const Napi::CallbackInfo &info) {
  torch::Tensor log2Tensor = tensor_.log2();
  return Tensor::New(info, log2Tensor);
}

Napi::Value Tensor::logit(const Napi::CallbackInfo &info) {
  torch::Tensor logitTensor = tensor_.logit();
  return Tensor::New(info, logitTensor);
}

Napi::Value Tensor::neg(const Napi::CallbackInfo &info) {
  torch::Tensor negTensor = tensor_.neg();
  return Tensor::New(info, negTensor);
}

Napi::Value Tensor::negative(const Napi::CallbackInfo &info) {
  torch::Tensor negativeTensor = tensor_.negative();
  return Tensor::New(info, negativeTensor);
}

Napi::Value Tensor::pin_memory(const Napi::CallbackInfo &info) {
  torch::Tensor pin_memoryTensor = tensor_.pin_memory();
  return Tensor::New(info, pin_memoryTensor);
}

Napi::Value Tensor::pinverse(const Napi::CallbackInfo &info) {
  torch::Tensor pinverseTensor = tensor_.pinverse();
  return Tensor::New(info, pinverseTensor);
}

Napi::Value Tensor::positive(const Napi::CallbackInfo &info) {
  torch::Tensor positiveTensor = tensor_.positive();
  return Tensor::New(info, positiveTensor);
}

Napi::Value Tensor::rad2deg(const Napi::CallbackInfo &info) {
  torch::Tensor rad2degTensor = tensor_.rad2deg();
  return Tensor::New(info, rad2degTensor);
}

Napi::Value Tensor::ravel(const Napi::CallbackInfo &info) {
  torch::Tensor ravelTensor = tensor_.ravel();
  return Tensor::New(info, ravelTensor);
}

Napi::Value Tensor::reciprocal(const Napi::CallbackInfo &info) {
  torch::Tensor reciprocalTensor = tensor_.reciprocal();
  return Tensor::New(info, reciprocalTensor);
}

Napi::Value Tensor::rsqrt(const Napi::CallbackInfo &info) {
  torch::Tensor rsqrtTensor = tensor_.rsqrt();
  return Tensor::New(info, rsqrtTensor);
}

Napi::Value Tensor::sigmoid(const Napi::CallbackInfo &info) {
  torch::Tensor sigmoidTensor = tensor_.sigmoid();
  return Tensor::New(info, sigmoidTensor);
}

Napi::Value Tensor::sign(const Napi::CallbackInfo &info) {
  torch::Tensor signTensor = tensor_.sign();
  return Tensor::New(info, signTensor);
}

Napi::Value Tensor::signbit(const Napi::CallbackInfo &info) {
  torch::Tensor signbitTensor = tensor_.signbit();
  return Tensor::New(info, signbitTensor);
}

Napi::Value Tensor::sgn(const Napi::CallbackInfo &info) {
  torch::Tensor sgnTensor = tensor_.sgn();
  return Tensor::New(info, sgnTensor);
}

Napi::Value Tensor::sin(const Napi::CallbackInfo &info) {
  torch::Tensor sinTensor = tensor_.sin();
  return Tensor::New(info, sinTensor);
}

Napi::Value Tensor::sinc(const Napi::CallbackInfo &info) {
  torch::Tensor sincTensor = tensor_.sinc();
  return Tensor::New(info, sincTensor);
}

Napi::Value Tensor::sinh(const Napi::CallbackInfo &info) {
  torch::Tensor sinhTensor = tensor_.sinh();
  return Tensor::New(info, sinhTensor);
}

Napi::Value Tensor::asinh(const Napi::CallbackInfo &info) {
  torch::Tensor asinhTensor = tensor_.asinh();
  return Tensor::New(info, asinhTensor);
}

Napi::Value Tensor::arcsinh(const Napi::CallbackInfo &info) {
  torch::Tensor arcsinhTensor = tensor_.arcsinh();
  return Tensor::New(info, arcsinhTensor);
}

Napi::Value Tensor::sqrt(const Napi::CallbackInfo &info) {
  torch::Tensor sqrtTensor = tensor_.sqrt();
  return Tensor::New(info, sqrtTensor);
}

Napi::Value Tensor::square(const Napi::CallbackInfo &info) {
  torch::Tensor squareTensor = tensor_.square();
  return Tensor::New(info, squareTensor);
}

Napi::Value Tensor::tan(const Napi::CallbackInfo &info) {
  torch::Tensor tanTensor = tensor_.tan();
  return Tensor::New(info, tanTensor);
}

Napi::Value Tensor::tanh(const Napi::CallbackInfo &info) {
  torch::Tensor tanhTensor = tensor_.tanh();
  return Tensor::New(info, tanhTensor);
}

Napi::Value Tensor::arctanh(const Napi::CallbackInfo &info) {
  torch::Tensor arctanhTensor = tensor_.arctanh();
  return Tensor::New(info, arctanhTensor);
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

Napi::Value Tensor::trace(const Napi::CallbackInfo &info) {
  torch::Tensor traceTensor = tensor_.trace();
  return Tensor::New(info, traceTensor);
}

Napi::Value Tensor::trunc(const Napi::CallbackInfo &info) {
  torch::Tensor truncTensor = tensor_.trunc();
  return Tensor::New(info, truncTensor);
}

Napi::Value Tensor::values(const Napi::CallbackInfo &info) {
  torch::Tensor valuesTensor = tensor_.values();
  return Tensor::New(info, valuesTensor);
}

} // namespace tensor

} // namespace jtorch
