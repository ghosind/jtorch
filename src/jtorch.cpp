#include "tensor/tensors.h"

using namespace jtorch;

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  tensor::Init(env, exports);

  return exports;
}

NODE_API_MODULE(jtorch, Init)
