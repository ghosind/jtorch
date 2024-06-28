#pragma once

#include "tensor.h"

#include <napi.h>

namespace jtorch {

namespace tensor {

Napi::Value isComplex(const Napi::CallbackInfo &);
Napi::Value isConj(const Napi::CallbackInfo &);
Napi::Value isFloatingPoint(const Napi::CallbackInfo &);
Napi::Value isNonzero(const Napi::CallbackInfo &);
Napi::Value isTensor(const Napi::CallbackInfo &);

Napi::Object Init(Napi::Env, Napi::Object);

} // namespace tensor

} // namespace jtorch
