// sherpa-onnx/csrc/log.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_LOG_H_
#define SHERPA_ONNX_CSRC_LOG_H_

#include <stdio.h>

#include <mutex>  // NOLINT
#include <sstream>
#include <string>

namespace sherpa_onnx {
class Voidifier {
};
#if !defined(SHERPA_ONNX_ENABLE_CHECK)
template <typename T>
const Voidifier &operator<<(const Voidifier &v, T &&) {
  return v;
}
#endif

}  // namespace sherpa_onnx

#define SHERPA_ONNX_STATIC_ASSERT(x) static_assert(x, "")

#ifdef SHERPA_ONNX_ENABLE_CHECK

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__PRETTY_FUNCTION__)
// for clang and GCC
#define SHERPA_ONNX_FUNC __PRETTY_FUNCTION__
#else
// for other compilers
#define SHERPA_ONNX_FUNC __func__
#endif

#define SHERPA_ONNX_CHECK(x)                                            \
  (x) ? (void)0                                                         \
      : ::sherpa_onnx::Voidifier() &                                    \
            ::sherpa_onnx::Logger(__FILE__, SHERPA_ONNX_FUNC, __LINE__, \
                                  ::sherpa_onnx::FATAL)                 \
                << "Check failed: " << #x << " "

// WARNING: x and y may be evaluated multiple times, but this happens only
// when the check fails. Since the program aborts if it fails, we don't think
// the extra evaluation of x and y matters.
//
// CAUTION: we recommend the following use case:
//
//      auto x = Foo();
//      auto y = Bar();
//      SHERPA_ONNX_CHECK_EQ(x, y) << "Some message";
//
//  And please avoid
//
//      SHERPA_ONNX_CHECK_EQ(Foo(), Bar());
//
//  if `Foo()` or `Bar()` causes some side effects, e.g., changing some
//  local static variables or global variables.
#define _SHERPA_ONNX_CHECK_OP(x, y, op)                                        \
  ((x)op(y)) ? (void)0                                                         \
             : ::sherpa_onnx::Voidifier() &                                    \
                   ::sherpa_onnx::Logger(__FILE__, SHERPA_ONNX_FUNC, __LINE__, \
                                         ::sherpa_onnx::FATAL)                 \
                       << "Check failed: " << #x << " " << #op << " " << #y    \
                       << " (" << (x) << " vs. " << (y) << ") "

#define SHERPA_ONNX_CHECK_EQ(x, y) _SHERPA_ONNX_CHECK_OP(x, y, ==)
#define SHERPA_ONNX_CHECK_NE(x, y) _SHERPA_ONNX_CHECK_OP(x, y, !=)
#define SHERPA_ONNX_CHECK_LT(x, y) _SHERPA_ONNX_CHECK_OP(x, y, <)
#define SHERPA_ONNX_CHECK_LE(x, y) _SHERPA_ONNX_CHECK_OP(x, y, <=)
#define SHERPA_ONNX_CHECK_GT(x, y) _SHERPA_ONNX_CHECK_OP(x, y, >)
#define SHERPA_ONNX_CHECK_GE(x, y) _SHERPA_ONNX_CHECK_OP(x, y, >=)

#define SHERPA_ONNX_LOG(x) \
  ::sherpa_onnx::Logger(__FILE__, SHERPA_ONNX_FUNC, __LINE__, ::sherpa_onnx::x)

// ------------------------------------------------------------
//       For debug check
// ------------------------------------------------------------
// If you define the macro "-D NDEBUG" while compiling kaldi-native-fbank,
// the following macros are in fact empty and does nothing.

#define SHERPA_ONNX_DCHECK(x) \
  ::sherpa_onnx::kDisableDebug ? (void)0 : SHERPA_ONNX_CHECK(x)

#define SHERPA_ONNX_DCHECK_EQ(x, y) \
  ::sherpa_onnx::kDisableDebug ? (void)0 : SHERPA_ONNX_CHECK_EQ(x, y)

#define SHERPA_ONNX_DCHECK_NE(x, y) \
  ::sherpa_onnx::kDisableDebug ? (void)0 : SHERPA_ONNX_CHECK_NE(x, y)

#define SHERPA_ONNX_DCHECK_LT(x, y) \
  ::sherpa_onnx::kDisableDebug ? (void)0 : SHERPA_ONNX_CHECK_LT(x, y)

#define SHERPA_ONNX_DCHECK_LE(x, y) \
  ::sherpa_onnx::kDisableDebug ? (void)0 : SHERPA_ONNX_CHECK_LE(x, y)

#define SHERPA_ONNX_DCHECK_GT(x, y) \
  ::sherpa_onnx::kDisableDebug ? (void)0 : SHERPA_ONNX_CHECK_GT(x, y)

#define SHERPA_ONNX_DCHECK_GE(x, y) \
  ::sherpa_onnx::kDisableDebug ? (void)0 : SHERPA_ONNX_CHECK_GE(x, y)

#define SHERPA_ONNX_DLOG(x)    \
  ::sherpa_onnx::kDisableDebug \
      ? (void)0                \
      : ::sherpa_onnx::Voidifier() & SHERPA_ONNX_LOG(x)

#else

#define SHERPA_ONNX_CHECK(x) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_LOG(x) ::sherpa_onnx::Voidifier()

#define SHERPA_ONNX_CHECK_EQ(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_NE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_LT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_LE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_GT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_GE(x, y) ::sherpa_onnx::Voidifier()

#define SHERPA_ONNX_DCHECK(x) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DLOG(x) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_EQ(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_NE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_LT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_LE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_GT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_GE(x, y) ::sherpa_onnx::Voidifier()

#endif  // SHERPA_ONNX_CHECK_NE

#endif  // SHERPA_ONNX_CSRC_LOG_H_
