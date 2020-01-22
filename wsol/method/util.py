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

__all__ = ['normalize_tensor', 'get_attention']


def normalize_tensor(x):
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor


def get_attention(feature, label):
    feat_map = feature.detach().clone()
    attention = feat_map.requires_grad_(True)[
                range(feat_map.size(0)), label, :, :]
    attention = attention.unsqueeze(1)
    attention = normalize_tensor(attention)
    return attention
