#pragma once

#include <napi.h>

namespace jtorch {

namespace tensor {

bool validateTensorParam(const Napi::CallbackInfo &);

} // namespace tensor

} // namespace jtorch
