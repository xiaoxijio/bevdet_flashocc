"""Training-only target transport; network forward and inference remain unchanged."""
from ..builder import DETECTORS
from .bevdet_det_occ import BEVDetDetOcc


@DETECTORS.register_module()
class BEVDetDetOccPrecomputedTargets(BEVDetDetOcc):
    def forward_train(self, det_targets=None, **kwargs):
        if det_targets is None:
            raise ValueError('This training recipe requires CenterHeadTargetsCPU in train_pipeline')
        self.pts_bbox_head._precomputed_targets = det_targets
        try:
            return super().forward_train(**kwargs)
        finally:
            self.pts_bbox_head._precomputed_targets = None
