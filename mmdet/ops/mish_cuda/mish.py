import torch  # Must import torch before C extension
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS
from mmcv.utils import TORCH_VERSION

from .mish_cuda_ext import mish_backward, mish_forward

if TORCH_VERSION != 'parrots' and TORCH_VERSION >= '1.6.0':
    from torch.cuda.amp import custom_bwd, custom_fwd
else:

    def custom_bwd(fun):
        return fun

    def custom_fwd(fun):
        return fun


class MishCudaFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inp):
        if not inp.is_contiguous():
            inp = inp.contiguous()
        ctx.save_for_backward(inp)
        return mish_forward(inp)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        if not ctx.needs_input_grad[0]:
            return (None, )
        return mish_backward(grad_out, inp)


class Mish(torch.nn.Module):

    def __init__(self, **kwargs):
        super(Mish, self).__init__()

    def forward(self, inp):
        return MishCudaFunction.apply(inp)


ACTIVATION_LAYERS.register_module(module=Mish)
