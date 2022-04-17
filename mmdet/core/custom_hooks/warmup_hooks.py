from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DetailedLinearWarmUpHook(Hook):

    def __init__(self,
                 warmup_iters=10000,
                 lr_weight_warmup_ratio=0.,
                 lr_bias_warmup_ratio=10.,
                 momentum_warmup_ratio=0.95):

        self.warmup_iters = warmup_iters
        self.lr_weight_warmup_ratio = lr_weight_warmup_ratio
        self.lr_bias_warmup_ratio = lr_bias_warmup_ratio
        self.momentum_warmup_ratio = momentum_warmup_ratio

        self.bias_base_lr = {}  # initial lr for all param groups
        self.weight_base_lr = {}
        self.base_momentum = {}

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if len(runner.optimizer.param_groups) != \
                len([*runner.model.parameters()]):
            runner.logger.warning(
                'optimizer config does not support preheat because'
                ' it is not using seperate param-group for each parameter')
            return

        for group_ind, (name,
                        param) in enumerate(runner.model.named_parameters()):
            group = runner.optimizer.param_groups[group_ind]
            self.base_momentum[group_ind] = group['momentum']
            if name.endswith('.bias'):
                self.bias_base_lr[group_ind] = group['lr']
            elif name.endswith('.weight'):
                self.weight_base_lr[group_ind] = group['lr']

    def before_train_iter(self, runner):
        if runner.iter <= self.warmup_iters:
            prog = runner.iter / self.warmup_iters
            for group_ind, bias_base in self.bias_base_lr.items():
                bias_warmup_lr = (
                    prog + (1 - prog) * self.lr_bias_warmup_ratio) * bias_base
                runner.optimizer.param_groups[group_ind]['lr'] = bias_warmup_lr
            for group_ind, weight_base in self.weight_base_lr.items():
                weight_warmup_lr = (
                    prog +
                    (1 - prog) * self.lr_weight_warmup_ratio) * weight_base
                runner.optimizer.param_groups[group_ind][
                    'lr'] = weight_warmup_lr
            for group_ind, momentum_base in self.base_momentum.items():
                warmup_momentum = (
                    prog +
                    (1 - prog) * self.momentum_warmup_ratio) * momentum_base
                runner.optimizer.param_groups[group_ind][
                    'momentum'] = warmup_momentum
