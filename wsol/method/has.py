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

import random

__all__ = ['has']


def has(image, grid_size, drop_rate):
    """
    Args:
        image: torch.Tensor, N x C x H x W, float32.
        grid_size: int
        drop_rate: float
    Returns:
        image: torch.Tensor, N x C x H x W, float32.
    """
    if grid_size == 0:
        return image

    batch_size, n_channels, height, width = image.size()

    for batch_idx in range(batch_size):
        for x in range(0, width, grid_size):
            for y in range(0, height, grid_size):
                x_end = min(height, x + grid_size)
                y_end = min(height, y + grid_size)
                if random.random() <= drop_rate:
                    image[batch_idx, :, x:x_end, y:y_end] = 0.
    return image