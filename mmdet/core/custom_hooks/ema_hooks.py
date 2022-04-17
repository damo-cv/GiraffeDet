import math

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import get_dist_info


@HOOKS.register_module()
class StateEMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointSaverHook.

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
        resume_from (str): The checkpoint path. Defaults to None.
    """

    def __init__(self,
                 momentum=0.9999,
                 interval=None,
                 nominal_batch_size=None,
                 warm_up=2000,
                 resume_from=None):
        self.interval = 1
        self.nominal_batch_size = None
        self.warm_up = warm_up
        if interval is not None:
            assert isinstance(interval, int) and interval > 0
            self.interval = interval
        elif nominal_batch_size is not None:
            self.interval = None
            self.nominal_batch_size = nominal_batch_size

        assert momentum > 0 and momentum < 1
        self.momentum = momentum
        self.checkpoint = resume_from

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_mapping = {}
        for name, value in model.state_dict().items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_mapping[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())

        if self.checkpoint is not None:
            runner.resume(self.checkpoint)
        else:
            cfg_text = runner.meta['config']
            cfg_dict = dict()
            exec(cfg_text, cfg_dict)
            checkpoint = cfg_dict['resume_from']
            if checkpoint is not None:
                runner.resume(checkpoint)

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        state_dict = model.state_dict()
        for name, buffer_name in self.param_ema_mapping.items():
            online_value = state_dict[name]
            ema_buffer = state_dict[buffer_name]

            momentum = self.momentum * (
                1 - math.exp(-runner.iter / (self.warm_up * self.interval)))

            if online_value.dtype.is_floating_point:
                ema_buffer.mul_(momentum).add_(
                    online_value.data.float(), alpha=1 - momentum)
            else:
                ema_buffer.data.copy_(online_value.data.float())

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters(runner)

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        if self.interval is None:
            assert self.nominal_batch_size is not None
            samples_per_gpu = runner.data_loader.sampler.samples_per_gpu
            _, word_size = get_dist_info()
            self.interval = math.ceil(self.nominal_batch_size /
                                      (samples_per_gpu * word_size))
        self._swap_ema_parameters(runner)

    def _swap_ema_parameters(self, runner):
        """Swap the parameter of model with parameter in ema_buffer."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        state_dict = model.state_dict()
        for name, buffer_name in self.param_ema_mapping.items():
            online_value = state_dict[name]
            online_dtype = online_value.dtype
            temp = online_value.data.clone().float()
            ema_buffer = state_dict[buffer_name]
            online_value.data.copy_(ema_buffer.data.to(online_dtype))
            # re-register buffer to force it to be fp32
            model.register_buffer(buffer_name, temp)
