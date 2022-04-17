#pragma once
#include <torch/types.h>

#ifdef __CUDACC__
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#define GLOBAL_INLINE __forceinline__ __host__ __device__
#else
#include <cmath>
#define GLOBAL_INLINE __inline__
#endif

#define THRESHOLD 20

// TODO: Try and convert these to lambda functions
template <typename scalar_t>
GLOBAL_INLINE scalar_t mish_fwd_func(scalar_t inp) {
  return inp * tanh(inp < scalar_t(THRESHOLD) ? log1p(exp(inp)) : inp);
};

template <typename scalar_t>
GLOBAL_INLINE scalar_t mish_bwd_func(scalar_t grad_out, scalar_t inp) {
  const scalar_t sp = inp < scalar_t(THRESHOLD) ? log1p(exp(inp)) : inp;
  const scalar_t grad_sp = 1 - exp(-sp);
  const scalar_t tsp = tanh(sp);
  const scalar_t grad_tsp = (1 - tsp * tsp) * grad_sp;
  const scalar_t grad = inp * grad_tsp + tsp;
  return grad_out * grad;
};

// Specialisations for Half to calculate as float
// Increases precision and also lacking certain instrinsics for Half
template <> GLOBAL_INLINE c10::Half mish_fwd_func(c10::Half inp) {
  return mish_fwd_func<float>((float)inp);
};

template <>
GLOBAL_INLINE c10::Half mish_bwd_func(c10::Half grad_out, c10::Half inp) {
  return mish_bwd_func<float>((float)grad_out, (float)inp);
};

template <> GLOBAL_INLINE c10::BFloat16 mish_fwd_func(c10::BFloat16 inp) {
  return mish_fwd_func<float>((float)inp);
};

template <>
GLOBAL_INLINE c10::BFloat16 mish_bwd_func(c10::BFloat16 grad_out,
                                          c10::BFloat16 inp) {
  return mish_bwd_func<float>((float)grad_out, (float)inp);
};