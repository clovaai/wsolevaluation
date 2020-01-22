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

from .util import get_attention

__all__ = ['AcolBase']


class AcolBase(nn.Module):
    def _acol_logits(self, feature, labels, drop_threshold):
        feat_map_a, logits = self._branch(feature=feature,
                                          classifier=self.classifier_A)
        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention = get_attention(feature=feat_map_a, label=labels)
        erased_feature = _erase_attention(
            feature=feature, attention=attention, drop_threshold=drop_threshold)
        feat_map_b, logit_b = self._branch(feature=erased_feature,
                                           classifier=self.classifier_B)
        return {'logits': logits, 'logit_b': logit_b,
                'feat_map_a': feat_map_a, 'feat_map_b': feat_map_b}

    def _branch(self, feature, classifier):
        feat_map = classifier(feature)
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.size(0), -1)
        return feat_map, logits


def _erase_attention(feature, attention, drop_threshold):
    b, _, h, w = attention.size()
    pos = torch.ge(attention, drop_threshold)
    mask = attention.new_ones((b, 1, h, w))
    mask[pos.data] = 0.
    erased_feature = feature * mask
    return erased_feature


def get_loss(output_dict, gt_labels, **kwargs):
    return nn.CrossEntropyLoss()(output_dict['logits'], gt_labels.long()) + \
           nn.CrossEntropyLoss()(output_dict['logit_b'], gt_labels.long())
