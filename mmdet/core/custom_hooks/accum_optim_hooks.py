import math

from mmcv.runner import HOOKS, Fp16OptimizerHook
from mmcv.runner.dist_utils import allreduce_grads, get_dist_info
from mmcv.utils import TORCH_VERSION

if TORCH_VERSION != 'parrots' and TORCH_VERSION >= '1.6.0':

    @HOOKS.register_module()
    class Fp16GradAccumulateOptimizerHook(Fp16OptimizerHook):

        def __init__(self, *wargs, **kwargs):
            nominal_batch_size = kwargs.pop('nominal_batch_size', None)
            accumulation = kwargs.pop('accumulation', None)
            self.accumulation = 1
            self.nominal_batch_size = None
            if accumulation is not None:
                assert isinstance(accumulation, int) and accumulation > 0
                self.accumulation = accumulation
            elif nominal_batch_size is not None:
                self.accumulation = None
                self.nominal_batch_size = nominal_batch_size

            super(Fp16GradAccumulateOptimizerHook,
                  self).__init__(*wargs, **kwargs)

        def before_train_epoch(self, runner):
            super(Fp16GradAccumulateOptimizerHook,
                  self).before_train_epoch(runner)
            if self.accumulation is None:
                assert self.nominal_batch_size is not None
                samples_per_gpu = runner.data_loader.sampler.samples_per_gpu
                _, word_size = get_dist_info()
                self.accumulation = math.ceil(self.nominal_batch_size /
                                              (samples_per_gpu * word_size))

        def after_train_iter(self, runner):
            # clear grads of last iteration
            if runner.iter % self.accumulation == 0:
                runner.model.zero_grad()
                runner.optimizer.zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()

            if (runner.iter + 1) % self.accumulation == 0:
                self.loss_scaler.unscale_(runner.optimizer)
                # grad clip
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters())
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {
                                'grad_norm': float(grad_norm),
                                'grad_scale': float(
                                    self.loss_scaler.get_scale())
                            }, runner.outputs['num_samples'])
                # step and update scaler
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)

else:

    @HOOKS.register_module()
    class Fp16GradAccumulateOptimizerHook(Fp16OptimizerHook):

        def __init__(self, *wargs, **kwargs):
            nominal_batch_size = kwargs.pop('nominal_batch_size', None)
            accumulation = kwargs.pop('accumulation', None)
            self.accumulation = 1
            self.nominal_batch_size = None
            if accumulation is not None:
                assert isinstance(accumulation, int) and accumulation > 0
                self.accumulation = accumulation
            elif nominal_batch_size is not None:
                self.accumulation = None
                self.nominal_batch_size = nominal_batch_size

            super(Fp16GradAccumulateOptimizerHook,
                  self).__init__(*wargs, **kwargs)

        def before_train_epoch(self, runner):
            super(Fp16GradAccumulateOptimizerHook,
                  self).before_train_epoch(runner)
            if self.accumulation is None:
                assert self.nominal_batch_size is not None
                samples_per_gpu = runner.data_loader.sampler.samples_per_gpu
                _, word_size = get_dist_info()
                self.accumulation = math.ceil(self.nominal_batch_size /
                                              (samples_per_gpu * word_size))

        def copy_grads_to_fp32(self, fp16_net, fp32_weights):
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights,
                                              fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(
                            fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)
                    # average gradients across accumulated iteration
                    fp32_param.grad /= self.accumulation

        def after_train_iter(self, runner):
            """Backward optimization steps for Mixed Precision Training.

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients (fp16).
            3. Copy gradients from the model to the fp32 weight copy.
            4. Scale the gradients back and update the fp32 weight copy.
            5. Copy back the params from fp32 weight copy to the fp16 model.
            """
            # clear grads of last iteration
            if runner.iter % self.accumulation == 0:
                runner.model.zero_grad()
                runner.optimizer.zero_grad()
            # scale the loss value
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()
            # copy fp16 grads in the model to fp32 params in the optimizer

            if (runner.iter + 1) % self.accumulation == 0:
                fp32_weights = []
                for param_group in runner.optimizer.param_groups:
                    fp32_weights += param_group['params']
                self.copy_grads_to_fp32(runner.model, fp32_weights)
                # allreduce grads
                if self.distributed:
                    allreduce_grads(fp32_weights, self.coalesce,
                                    self.bucket_size_mb)

                has_overflow = self.loss_scaler.has_overflow(fp32_weights)
                # if has overflow, skip this iteration
                if not has_overflow:
                    # scale the gradients back
                    for param in fp32_weights:
                        if param.grad is not None:
                            param.grad.div_(self.loss_scaler.loss_scale)
                    if self.grad_clip is not None:
                        grad_norm = self.clip_grads(fp32_weights)
                        if grad_norm is not None:
                            # Add grad norm to the logger
                            runner.log_buffer.update(
                                {'grad_norm': float(grad_norm)},
                                runner.outputs['num_samples'])
                    # update fp32 params
                    runner.optimizer.step()
                    # copy fp32 params to the fp16 model
                    self.copy_params_to_fp16(runner.model, fp32_weights)
                self.loss_scaler.update_scale(has_overflow)
                if has_overflow:
                    runner.logger.warning(
                        'Check overflow, downscale loss scale '
                        f'to {self.loss_scaler.cur_scale}')
