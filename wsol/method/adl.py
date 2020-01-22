"""
Original repository: https://github.com/junsukchoe/ADL
"""

import torch
import torch.nn as nn

__all__ = ['ADL']


class ADL(nn.Module):
    def __init__(self, adl_drop_rate=0.75, adl_drop_threshold=0.8):
        super(ADL, self).__init__()
        if not (0 <= adl_drop_rate <= 1):
            raise ValueError("Drop rate must be in range [0, 1].")
        if not (0 <= adl_drop_threshold <= 1):
            raise ValueError("Drop threshold must be in range [0, 1].")
        self.adl_drop_rate = adl_drop_rate
        self.adl_drop_threshold = adl_drop_threshold
        self.attention = None
        self.drop_mask = None

    def forward(self, input_):
        if not self.training:
            return input_
        else:
            attention = torch.mean(input_, dim=1, keepdim=True)
            importance_map = torch.sigmoid(attention)
            drop_mask = self._drop_mask(attention)
            selected_map = self._select_map(importance_map, drop_mask)
            return input_.mul(selected_map)

    def _select_map(self, importance_map, drop_mask):
        random_tensor = torch.rand([], dtype=torch.float32) + self.adl_drop_rate
        binary_tensor = random_tensor.floor()
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    def _drop_mask(self, attention):
        b_size = attention.size(0)
        max_val, _ = torch.max(attention.view(b_size, -1), dim=1, keepdim=True)
        thr_val = max_val * self.adl_drop_threshold
        thr_val = thr_val.view(b_size, 1, 1, 1)
        return (attention < thr_val).float()

    def extra_repr(self):
        return 'adl_drop_rate={}, adl_drop_threshold={}'.format(
            self.adl_drop_rate, self.adl_drop_threshold)
