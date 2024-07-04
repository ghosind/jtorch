#pragma once

#define THROW_ERROR(env, msg) Napi::Error::New(env, msg).ThrowAsJavaScriptException()
#define THROW_TYPE_ERROR(env, msg) Napi::TypeError::New(env, msg).ThrowAsJavaScriptException()

#define BOOLEAN_VALUE(val) val.ToBoolean().Value()
#define INT32_VALUE(val) val.ToNumber().Int32Value()
#define UINT32_VALUE(val) val.ToNumber().Uint32Value()
#define INT64_VALUE(val) val.ToNumber().Int64Value()
#define FLOAT_VALUE(val) val.ToNumber().FloatValue()
#define DOUBLE_VALUE(val) val.ToNumber().DoubleValue()
#define STRING_VALUE(val) val.ToString().Utf8Value()
