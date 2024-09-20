import os
import math
import torch
from torch import Tensor
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleTransformerTagger = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer.py'), 'ParT').ParticleTransformerTagger


def get_model(data_config, **kwargs):

    cfg = dict(
        pf_input_dim=len(data_config.input_dicts['pf_features']),
        sv_input_dim=len(data_config.input_dicts['sv_features']),
        num_classes=len(data_config.label_value) + 3*len(data_config.label_value_custom), # dim_cls + 3*dim_reg
        # network configurations
        pair_input_dim=4,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    #kwargs.pop('loss_gamma')
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerTagger(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


class LogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', baseline=0.) -> None:
        super(LogCoshLoss, self).__init__(None, None, reduction)
        self.baseline = baseline

    def forward(self, input: Tensor, target: Tensor, use_input: Tensor) -> Tensor:
        x = (input - target)
        loss = (x + torch.nn.functional.softplus(-2. * x) - math.log(2)) * use_input + self.baseline * (~use_input)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        

class QuantileLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', quantile=0.16, baseline=0.) -> None:
        super(QuantileLoss, self).__init__(None, None, reduction)
        self.quantile = quantile
        self.baseline = baseline

    def forward(self, input: Tensor, target: Tensor, use_input: Tensor) -> Tensor:
        z = (target - input)
        loss = (self.quantile * z * (z>=0) + (self.quantile - 1) * z * (z<0)) * use_input + self.baseline * (~use_input)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class HybridLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', factor_reg=1., factor_err=1., baseline_reg=0., baseline_err=0.) -> None:
        super(HybridLoss, self).__init__(None, None, reduction)
        self.loss_cls_fn = torch.nn.CrossEntropyLoss()
        self.loss_reg_fn = LogCoshLoss(baseline=baseline_reg)
        self.loss_err_fn_minus = QuantileLoss(quantile=0.16, baseline=baseline_err/2)
        self.loss_err_fn_plus = QuantileLoss(quantile=0.84, baseline=baseline_err/2)
        self.factor_reg = factor_reg
        self.factor_err = factor_err

    def forward(self, input_cls: Tensor, input_reg: Tensor, input_err_plus: Tensor, input_err_minus: Tensor, target_cls: Tensor, target_reg: Tensor) -> Tensor:
        loss_cls = self.loss_cls_fn(input_cls, target_cls)
        loss_reg = self.loss_reg_fn(input_reg, target_reg, (target_cls <= 1))
        loss_err = self.loss_err_fn_plus(input_err_plus, target_reg, (target_cls <= 1)) + self.loss_err_fn_minus(input_err_minus, target_reg, (target_cls <= 1))
        loss = loss_cls + self.factor_reg * loss_reg + self.factor_err * loss_err
        return loss, {'cls': loss_cls.item(), 'reg': loss_reg.item(), 'err': loss_err.item()}


def get_loss(data_config, **kwargs):
    factor_reg = kwargs.get('factor_reg', 1)
    factor_err = kwargs.get('factor_err', 1)
    baseline_reg = kwargs.get('baseline_reg', 0.)
    baseline_err = kwargs.get('baseline_err', 0.)
    return HybridLoss(factor_reg=factor_reg, factor_err=factor_err, baseline_reg=baseline_reg, baseline_err=baseline_err)
