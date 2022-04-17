#include <torch/types.h>

namespace mish_cpu_kernel {
#include "../mish.h"

void mish(at::Tensor inp, at::Tensor out) {
  AT_DISPATCH_ALL_TYPES(inp.scalar_type(), "mish_cpu_kernel", [&]() {
    scalar_t *inp_ptr = inp.data_ptr<scalar_t>();
    scalar_t *out_ptr = out.data_ptr<scalar_t>();

    scalar_t *inp_ptr_end = inp_ptr + inp.numel();
    for (; inp_ptr != inp_ptr_end; inp_ptr++, out_ptr++)
      *out_ptr = mish_fwd_func<scalar_t>(*inp_ptr);
  });
}

void mish_backward(at::Tensor grad_out, at::Tensor inp, at::Tensor grad_inp) {
  AT_DISPATCH_ALL_TYPES(
      grad_out.scalar_type(), "mish_backward_cpu_kernel", [&]() {
        scalar_t *grad_out_ptr = grad_out.data_ptr<scalar_t>();
        scalar_t *inp_ptr = inp.data_ptr<scalar_t>();
        scalar_t *grad_inp_ptr = grad_inp.data_ptr<scalar_t>();

        scalar_t *grad_out_ptr_end = grad_out_ptr + grad_out.numel();
        for (; grad_out_ptr != grad_out_ptr_end;
             grad_out_ptr++, inp_ptr++, grad_inp_ptr++)
          *grad_inp_ptr = mish_bwd_func<scalar_t>(*grad_out_ptr, *inp_ptr);
      });
}

} // namespace mish_cpu_kernel
