"""SDK EfficientNetB0HZ topology, independent of torchvision ImageNet models.

Block definitions match SDK FlashOCC_Board/efficientnet_no_se.py:
ds 16x1; ir 24x2/40x2/80x3/112x3/192x4/320x1, no SE, ReLU6.
SDK ONNX folded Conv/BN weights are imported by convert_g5_official_onnx.py.
"""
from torch import nn
from mmcv.runner import BaseModule, HOOKS, Hook
from ..builder import BACKBONES


class _SDKBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, expansion):
        super().__init__()
        mid = in_channels * expansion
        self.expansion = expansion
        self.has_skip = stride == 1 and in_channels == out_channels
        if expansion != 1:
            self.conv_pw = nn.Conv2d(in_channels, mid, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid)
            self.conv_dw = nn.Conv2d(mid, mid, kernel, stride,
                                     kernel // 2, groups=mid, bias=False)
            self.bn2 = nn.BatchNorm2d(mid)
            self.conv_pwl = nn.Conv2d(mid, out_channels, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel, stride,
                                     kernel // 2, groups=in_channels, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv_pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=False)

    def forward(self, x):
        identity = x
        if self.expansion != 1:
            x = self.act(self.bn1(self.conv_pw(x)))
            x = self.act(self.bn2(self.conv_dw(x)))
            x = self.bn3(self.conv_pwl(x))
        else:
            x = self.act(self.bn1(self.conv_dw(x)))
            x = self.bn2(self.conv_pw(x))
        return x + identity if self.has_skip else x


@BACKBONES.register_module()
class EfficientNetB0HZ(BaseModule):
    def __init__(self, pretrained=None, norm_eval=True):
        super().__init__()
        if pretrained is not None:
            raise ValueError('Use the converted full G5 load_from checkpoint; '
                             'torchvision weights are not SDK no-SE pretraining')
        self.norm_eval = norm_eval
        self.conv_stem = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act = nn.ReLU6(inplace=False)
        in_channels = 32
        spec = [(16, 1, 3, 1, 1), (24, 2, 3, 2, 6), (40, 2, 5, 2, 6),
                (80, 3, 3, 2, 6), (112, 3, 5, 1, 6),
                (192, 4, 5, 2, 6), (320, 1, 3, 1, 6)]
        for stage, (out, repeat, kernel, stride, expansion) in enumerate(spec):
            blocks = []
            for index in range(repeat):
                blocks.append(_SDKBlock(in_channels, out, kernel,
                                        stride if index == 0 else 1, expansion))
                in_channels = out
            setattr(self, 'block_' + str(stage), nn.Sequential(*blocks))
        self.conv_head = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

    def forward(self, x):
        x = self.act(self.bn1(self.conv_stem(x)))
        outputs = []
        for stage in range(7):
            x = getattr(self, 'block_' + str(stage))(x)
            if stage == 4:
                outputs.append(x)
        outputs.append(self.act(self.bn2(self.conv_head(x))))
        return tuple(outputs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


@HOOKS.register_module()
class G5FoldedBatchNormHook(Hook):
    """Retain ONNX-recovered affine normalization; gamma/beta remain trainable.

    Original pre-fusion BN moments are unrecoverable. Resetting running moments
    in training would invalidate the verified SDK initialization.
    """
    def before_train_iter(self, runner):
        for module in runner.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
