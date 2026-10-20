"""Torchvision B0 feature extractor without SE, using ReLU6 for deployment.

Follows the SDK's backbone choice and output shapes, but uses torchvision keys;
SDK EfficientNetB0HZ checkpoints are NOT interchangeable with this module.
"""
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision.ops.misc import SqueezeExcitation
from mmcv.runner import BaseModule, _load_checkpoint
from ..builder import BACKBONES


@BACKBONES.register_module()
class EfficientNetB0NoSE(BaseModule):
    def __init__(self, pretrained=None, norm_eval=False):
        super().__init__()
        self.pretrained = pretrained
        self.norm_eval = norm_eval
        self.features = efficientnet_b0(weights=None, stochastic_depth_prob=0.).features
        self._replace_deployment_layers(self.features)

    @classmethod
    def _replace_deployment_layers(cls, module):
        for name, child in list(module.named_children()):
            if isinstance(child, SqueezeExcitation):
                module.add_module(name, nn.Identity())
            elif isinstance(child, nn.SiLU):
                module.add_module(name, nn.ReLU6(inplace=True))
            else:
                cls._replace_deployment_layers(child)

    def init_weights(self):
        if self._is_init:
            return
        if self.pretrained is not None:
            # Load all retained ImageNet features; only SE/classifier tensors
            # may be discarded. Shape/key mismatches in retained layers fail.
            raw = _load_checkpoint(self.pretrained, map_location='cpu')
            raw = raw.get('state_dict', raw)
            source = {}
            for key, value in raw.items():
                key = key[7:] if key.startswith('module.') else key
                if key.startswith('img_backbone.'):
                    key = key[len('img_backbone.'):]
                source[key] = value
            target = self.state_dict()
            missing = [k for k, v in target.items()
                       if k not in source or source[k].shape != v.shape]
            if missing:
                raise ValueError('Incompatible torchvision B0 weights: ' + ', '.join(missing[:10]))
            self.load_state_dict({k: source[k] for k in target}, strict=True)
        self._is_init = True

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (5, 8):
                outputs.append(x)
        return tuple(outputs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self
