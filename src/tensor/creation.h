#pragma once

#include <napi.h>

namespace jtorch {

namespace tensor {

Napi::Value arange(const Napi::CallbackInfo &);

} // namespace tensor

} // namespace jtorch
