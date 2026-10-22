"""Fixed-capacity runtime maps for BEV pooling; no geometry or scatter in ONNX."""
import math
import numpy as np
import torch


def make_layout(metas, grid_size, channels, reserve=1.5):
    if not math.isfinite(reserve) or reserve < 1:
        raise ValueError('mapping reserve must be >= 1')
    if int(grid_size[2]) != 1 or metas[0] is None:
        raise ValueError('Layout requires valid sample geometry and one BEV Z bin')
    lengths = metas[4].detach().cpu().numpy()
    bound, groups = 1, []
    while bound < 2 * int(lengths.max()):
        count = int(((lengths <= bound) & (lengths > (bound // 2 if bound > 1 else 0))).sum())
        # Keep even empty buckets available for a different rig.
        capacity = max(8, math.ceil(count * reserve))
        if capacity > 65535 or bound > 65535:
            raise ValueError('Mapping bucket dimension exceeds SDK limit; use a smaller reserve or a tiled layout')
        groups.append(dict(bound=bound, count=capacity))
        bound *= 2
    return dict(version=2, index_encoding='base1024_fp16_safe', height=int(grid_size[1]), width=int(grid_size[0]),
                channels=int(channels), groups=groups)


def encode_indices(array):
    # SDK Binary inputs do not accept I32. Two exact FP16-safe base-1024
    # components avoid rounding large indices when inputs are kept in FP16.
    array = np.asarray(array, dtype=np.int64)
    if np.any(array < 0) or np.any(array >= 1024 * 1024):
        raise ValueError('Mapping index exceeds exact base-1024 encoding capacity')
    return np.stack((array % 1024, array // 1024)).astype(np.float32)


def decode_indices(packed):
    # Keep bucket axes until after Cast/Mul/Add: flattened lengths >65535
    # make this SDK select software conversion kernels that cannot be packed.
    low, high = packed.unbind(0)
    return (low.to(torch.int32) + high.to(torch.int32) * 1024).reshape(-1)


def make_mapping(metas, layout, device='cpu'):
    h, w = layout['height'], layout['width']
    lookup = np.zeros(h * w, dtype=np.int32)
    result = []
    if metas[0] is None:
        rb = rd = rf = starts = lengths = np.array([], dtype=np.int64)
    else:
        rb, rd, rf, starts, lengths = [t.detach().cpu().numpy().astype(np.int64) for t in metas]
    if len(lengths) and lengths.max() > layout['groups'][-1]['bound']:
        raise ValueError('Mapping exceeds maximum bucket bound; export a larger shared layout')
    offset = 1  # slot zero is the constant all-zero BEV feature
    for group in layout['groups']:
        bound, count = group['bound'], group['count']
        ids = np.flatnonzero((lengths <= bound) & (lengths > (bound // 2 if bound > 1 else 0)))
        if len(ids) > count:
            raise ValueError(f'Mapping bucket {bound} needs {len(ids)} slots, compiled capacity is {count}; no points were truncated')
        di = np.zeros((count, bound), dtype=np.int32)
        fi = np.zeros_like(di)
        mask = np.zeros((count, bound, 1), dtype=np.float32)
        for slot, index in enumerate(ids):
            start, length = starts[index], lengths[index]
            di[slot, :length] = rd[start:start + length]
            fi[slot, :length] = rf[start:start + length]
            mask[slot, :length, 0] = 1
            voxel = int(rb[start])
            if not 0 <= voxel < h * w:
                raise ValueError('BEV index outside shared layout')
            lookup[voxel] = offset + slot
        result.extend([torch.from_numpy(encode_indices(di)).to(device),
                       torch.from_numpy(encode_indices(fi)).to(device),
                       torch.from_numpy(mask).to(device)])
        offset += count
    result.append(torch.from_numpy(encode_indices(lookup)).to(device))
    return tuple(result)


def mapping_names(layout):
    names = []
    for group in layout['groups']:
        names.extend(f'{kind}_{group["bound"]}' for kind in ('ranks_depth', 'ranks_feat', 'pool_mask'))
    return names + ['bev_slots']


class RuntimeMappingPool(torch.nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.channels = layout['channels']

    def forward(self, depth, feat, *mapping):
        depth = depth.reshape(-1, 1)
        feat = feat.reshape(-1, self.channels)
        sums = [feat.new_zeros((1, self.channels))]
        for i, group in enumerate(self.layout['groups']):
            rd, rf, mask = mapping[3*i:3*i+3]
            rd, rf = decode_indices(rd), decode_indices(rf)
            values = torch.index_select(depth, 0, rd) * torch.index_select(feat, 0, rf)
            values = values.reshape(group['count'], group['bound'], self.channels) * mask
            sums.append(values.sum(1).reshape(group['count'], self.channels))
        bev = torch.index_select(torch.cat(sums, 0), 0, decode_indices(mapping[-1]))
        return bev.reshape(1, self.layout['height'], self.layout['width'], self.channels).permute(0, 3, 1, 2).contiguous()
