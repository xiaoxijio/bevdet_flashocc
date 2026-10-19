# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.nn import functional as F
from ..builder import NECKS


@NECKS.register_module()
class BEVGridAdapter(nn.Module):
    """Resample BEV cell centers in metric coordinates (B,C,Y,X).

    Both grids are [lower, upper, step], shared by x and y. This preserves
    the detection encoder and never creates a second BEV encoder.
    """
    def __init__(self, src_grid, dst_grid, in_channels=256, out_channels=128):
        super().__init__()
        self.src_size = round((src_grid[1] - src_grid[0]) / src_grid[2])
        dst_size = round((dst_grid[1] - dst_grid[0]) / dst_grid[2])
        centers = dst_grid[0] + (torch.arange(dst_size).float() + .5) * dst_grid[2]
        # align_corners=False: -1/+1 denote the source grid outer edges.
        normalized = 2 * (centers - src_grid[0]) / (self.src_size * src_grid[2]) - 1
        yy, xx = torch.meshgrid(normalized, normalized, indexing='ij')
        self.register_buffer('sample_grid', torch.stack((xx, yy), -1)[None],
                             persistent=False)
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.ReLU())

    def forward(self, x):
        if (not torch.onnx.is_in_onnx_export()
                and x.shape[-2:] != (self.src_size, self.src_size)):
            raise ValueError('BEV feature shape does not match src_grid')
        grid = self.sample_grid.to(x).expand(x.shape[0], -1, -1, -1)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                          align_corners=False)
        return self.proj(x)
