"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
