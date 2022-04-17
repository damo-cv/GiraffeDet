#include <torch/extension.h>

using namespace pybind11::literals;

namespace mish_cuda_kernel {
void mish(at::Tensor inp, at::Tensor out);
void mish_backward(at::Tensor grad_out, at::Tensor inp, at::Tensor grad_inp);
} // namespace mish_cuda_kernel
namespace mish_cpu_kernel {
void mish(at::Tensor inp, at::Tensor out);
void mish_backward(at::Tensor grad_out, at::Tensor inp, at::Tensor grad_inp);
} // namespace mish_cpu_kernel

torch::Tensor mish_forward(const torch::Tensor &input) {
  auto output = torch::empty_like(input);
  if (input.is_cuda()) {
    mish_cuda_kernel::mish(input, output);
  } else {
    mish_cpu_kernel::mish(input, output);
  }
  return output;
}

torch::Tensor mish_backward(const torch::Tensor &grad_out,
                            const torch::Tensor &input) {
  torch::Tensor grad_inp = torch::empty_like(input);
  if (grad_out.is_cuda()) {
    mish_cuda_kernel::mish_backward(grad_out, input, grad_inp);
  } else {
    mish_cpu_kernel::mish_backward(grad_out, input, grad_inp);
  }
  return grad_inp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mish_forward", &mish_forward, "Mish activation forward", "input"_a);
  m.def("mish_backward", &mish_backward, "Mish activation backward",
        "grad_out"_a, "input"_a);
}
