#include "tensor/tensor.h"

using namespace jtorch;

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Tensor::Init(env, exports);

  return exports;
}

NODE_API_MODULE(jtorch, Init)
