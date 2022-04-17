#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

// TORCH_CHECK replaces AT_CHECK in PyTorch 1,2, support 1.1 as well.
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

namespace mish_cuda_kernel {
#include "../mish.h"

const size_t threadsPerBlock = 512;
const size_t maxGridDim = 50000;

template <typename scalar_t>
__global__ void mish_cuda_kernel(scalar_t *inp, scalar_t *out, size_t numel) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel;
       tid += gridDim.x * blockDim.x)
    *(out + tid) = mish_fwd_func<scalar_t>(*(inp + tid));
}

template <typename scalar_t>
__global__ void mish_backward_cuda_kernel(scalar_t *grad_out, scalar_t *inp,
                                          scalar_t *grad_inp, size_t numel) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel;
       tid += gridDim.x * blockDim.x)
    *(grad_inp + tid) =
        mish_bwd_func<scalar_t>(*(grad_out + tid), *(inp + tid));
}

void mish(at::Tensor inp, at::Tensor out) {
  CHECK_INPUT(inp);
  CHECK_INPUT(out);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, inp.scalar_type(),
      "mish_cuda_kernel", [&]() {
        const size_t numel = inp.numel();
        dim3 blocks(std::min(maxGridDim,
                             at::cuda::ATenCeilDiv(numel, threadsPerBlock)));
        dim3 threads(threadsPerBlock);
        mish_cuda_kernel<<<blocks, threads>>>(inp.data_ptr<scalar_t>(),
                                              out.data_ptr<scalar_t>(), numel);
      });
}

void mish_backward(at::Tensor grad_out, at::Tensor inp, at::Tensor grad_inp) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(inp);
  CHECK_INPUT(grad_inp);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, grad_out.scalar_type(),
      "mish_backward_cuda_kernel", [&]() {
        const size_t numel = grad_out.numel();
        dim3 blocks(std::min(maxGridDim,
                             at::cuda::ATenCeilDiv(numel, threadsPerBlock)));
        dim3 threads(threadsPerBlock);
        mish_backward_cuda_kernel<<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(), inp.data_ptr<scalar_t>(),
            grad_inp.data_ptr<scalar_t>(), numel);
      });
}

} // namespace mish_cuda_kernel
