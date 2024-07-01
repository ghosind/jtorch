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
    InstanceMethod("conjPhysical", &Tensor::conj_physical),
    InstanceMethod("resolveConj", &Tensor::resolve_conj),
    InstanceMethod("resolveNeg", &Tensor::resolve_neg),
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
    InstanceMethod("intRepr", &Tensor::int_repr),
    InstanceMethod("inverse", &Tensor::inverse),
    InstanceMethod("isfinite", &Tensor::isfinite),
    InstanceMethod("isinf", &Tensor::isinf),
    InstanceMethod("isposinf", &Tensor::isposinf),
    InstanceMethod("isneginf", &Tensor::isneginf),
    InstanceMethod("isnan", &Tensor::isnan),
    InstanceMethod("isComplex", &Tensor::is_complex),
    InstanceMethod("isConj", &Tensor::is_conj),
    InstanceMethod("isFloatingPoint", &Tensor::is_floating_point),
    InstanceMethod("isNonzero", &Tensor::is_nonzero),
    InstanceMethod("lgamma", &Tensor::lgamma),
    InstanceMethod("log", &Tensor::log),
    InstanceMethod("logdet", &Tensor::logdet),
    InstanceMethod("log10", &Tensor::log10),
    InstanceMethod("log1p", &Tensor::log1p),
    InstanceMethod("log2", &Tensor::log2),
    InstanceMethod("logit", &Tensor::logit),
    InstanceMethod("neg", &Tensor::neg),
    InstanceMethod("negative", &Tensor::negative),
    InstanceMethod("pinMemory", &Tensor::pin_memory),
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

Napi::Value Tensor::toString(const Napi::CallbackInfo &info) {
  // TODO: more clear output

  std::ostringstream stream;
  stream << tensor_;
  std::string str = stream.str();
  return Napi::String::New(info.Env(), str);
}

} // namespace tensor

} // namespace jtorch
