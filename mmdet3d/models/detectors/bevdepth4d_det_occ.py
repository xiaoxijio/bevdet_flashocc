"""Joint temporal detection/FlashOCC; one shared multi-frame BEV encoder."""
from ..builder import DETECTORS
from .bevdet import BEVDepth4D
from .bevdet_det_occ import BEVDetDetOcc


@DETECTORS.register_module()
class BEVDepth4DDetOcc(BEVDetDetOcc, BEVDepth4D):
    """Reuse joint losses/inference and BEVDepth4D temporal feature extraction.

    Cooperative initialization reaches BEVDet4D through the MRO. Historical
    image features are computed without gradients, as in BEVDepth4D.
    """
    def forward_train(self, det_targets=None, **kwargs):
        if det_targets is None:
            raise ValueError('CenterHeadTargetsCPU must supply det_targets')
        self.pts_bbox_head._precomputed_targets = det_targets
        try:
            return super().forward_train(**kwargs)
        finally:
            self.pts_bbox_head._precomputed_targets = None


from mmcv.runner import HOOKS, Hook
from torch import nn

@HOOKS.register_module()
class D1BackboneBatchNormHook(Hook):
    """Preserve SDK folded backbone BN; downstream BN learns temporal features."""
    def before_train_iter(self, runner):
        model = runner.model
        while hasattr(model, 'module'):
            model = model.module
        for layer in model.img_backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
