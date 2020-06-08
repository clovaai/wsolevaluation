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

import numpy as np
import os
import unittest

from data_loaders import configure_metadata
from data_loaders import get_image_ids
from evaluation import BoxEvaluator
from evaluation import calculate_multiple_iou
from evaluation import compute_bboxes_from_scoremaps
from evaluation import get_mask
from evaluation import MaskEvaluator
from evaluation import resize_bbox


class EvalUtilTest(unittest.TestCase):
    def test_calculate_multiple_iou_shape_type_degenerate_zero(self):
        box_a = np.zeros((3, 4), dtype=np.int)
        box_b = np.zeros((5, 4), dtype=np.int)
        ious = calculate_multiple_iou(box_a, box_b)
        self.assertIsInstance(ious, np.ndarray)
        self.assertEqual(ious.dtype, np.float)
        self.assertEqual(ious.shape, (3, 5))
        for i in range(ious.shape[0]):
            for j in range(ious.shape[1]):
                self.assertEqual(ious[i, j], 1.0)

    def test_calculate_multiple_iou_box_convention(self):
        box_a = np.array([
            [1, 1, 0, 0],
        ], dtype=np.int)
        box_b = np.array([
            [1, 1, 2, 2],
        ], dtype=np.int)
        self.assertRaises(RuntimeError, calculate_multiple_iou, box_a, box_b)

    def test_calculate_multiple_iou_eye_boxes(self):
        box_a = np.array([
            [1, 1, 3, 3],
            [4, 4, 5, 5],
        ], dtype=np.int)
        box_b = np.array([
            [1, 1, 3, 3],
            [4, 4, 5, 5],
        ], dtype=np.int)
        ious = calculate_multiple_iou(box_a, box_b)
        self.assertTrue((ious == np.eye(2)).all())

    def test_calculate_multiple_iou_realistic_values(self):
        box_a = np.array([
            [1, 2, 5, 6],
        ], dtype=np.int)
        box_b = np.array([
            [3, 1, 4, 5],
        ], dtype=np.int)
        ious = calculate_multiple_iou(box_a, box_b)
        self.assertAlmostEqual(ious[0, 0], 8. / 27, delta=1e-6)

    def test_resize_bbox_case(self):
        box = 1, 2, 4, 3
        image_size = 10, 20
        resize_size = 35, 10
        resized_box = resize_bbox(box, image_size, resize_size)
        self.assertEqual(resized_box, (3, 1, 14, 1))

    def test_resize_bbox_degenerate(self):
        box = 1, 2, 1, 2
        image_size = 3, 3
        resize_size = 3, 3
        resized_box = resize_bbox(box, image_size, resize_size)
        self.assertEqual(resized_box, (1, 2, 1, 2))

    def test_compute_bboxes_from_scoremaps_degenerate(self):
        scoremap = np.zeros([3, 3], dtype=np.float)
        scoremap_threshold_list = np.arange(0, 1, 0.2)
        boxes = compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list)[0]
        boxes = [box[0].tolist() for box in boxes]
        self.assertListEqual(boxes, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                     [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_compute_bboxes_from_scoremaps_unimodal(self):
        # array has shape array[y][x]
        scoremap = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.4, 0.4, 0.4, 0.6],
                             [0.0, 0.4, 1.0, 0.8, 0.6],
                             [0.0, 0.4, 0.4, 0.4, 0.6]],
                            dtype=np.float)
        scoremap_threshold_list = np.arange(0, 1, 0.2)
        boxes = compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list)[0]
        boxes = [box[0].tolist() for box in boxes]
        self.assertListEqual(boxes, [[1, 1, 4, 3],
                                     [1, 1, 4, 3],
                                     [2, 1, 4, 3],
                                     [2, 2, 4, 3],
                                     [2, 2, 3, 3]])

    def test_compute_bboxes_from_scoremaps_multimodal(self):
        # array has shape array[y][x]
        scoremap = np.array([[0.4, 0.0, 0.2, 0.2, 0.2],
                             [0.4, 0.4, 0.0, 0.4, 0.0],
                             [0.0, 0.0, 0.0, 0.4, 0.0],
                             [1.0, 0.6, 0.8, 0.2, 0.2]],
                            dtype=np.float)
        scoremap_threshold_list = np.arange(0, 1, 0.2)
        boxes = compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list)[0]
        boxes = [box[0].tolist() for box in boxes]
        self.assertListEqual(boxes, [[0, 0, 4, 3],
                                     [0, 0, 2, 2],
                                     [0, 3, 3, 3],
                                     [2, 3, 3, 3],
                                     [0, 3, 1, 3]])

    def test_compute_bboxes_from_scoremaps_nan(self):
        # array has shape array[y][x]
        scoremap = np.full([3, 3], np.nan)
        scoremap_threshold_list = np.arange(0, 1, 0.2)
        self.assertRaises(ValueError, compute_bboxes_from_scoremaps,
                          scoremap, scoremap_threshold_list)


def set_metadata(dataset_name, split):
    metadata_root = os.path.join('metadata', dataset_name, split)
    metadata = configure_metadata(metadata_root)
    return metadata


def load_evaluator(Evaluator, dataset_name, split, cam_curve_interval,
                   mask_root=None):
    metadata = set_metadata(dataset_name, split)
    cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
    iou_threshold_list = [30, 50, 70]
    evaluator = Evaluator(metadata=metadata,
                          dataset_name=dataset_name,
                          split=split,
                          cam_threshold_list=cam_threshold_list,
                          iou_threshold_list=iou_threshold_list,
                          multi_contour_eval=False,
                          mask_root=mask_root)
    return evaluator


class BoxEvaluatorTest(unittest.TestCase):
    _DATASET_NAMES = ('CUB', 'ILSVRC')
    _SPLITS = ('val', 'test')
    _CAM_CURVE_INTERVAL = 0.01
    _BOX_VERIFICATION = {
        'CUB': {
            'val': [
                [(176, 136, 459, 291)],
                [(44, 183, 426, 254)],
                [(437, 42, 834, 422)],
            ],
            'test': [
                [(60, 27, 385, 331)],
                [(14, 112, 402, 298)],
                [(33, 53, 284, 448)],
            ],
        },
        'ILSVRC': {
            'val': [
                [(156, 163, 318, 230)],
                [(143, 142, 394, 248)],
                [(28, 94, 485, 303)],
            ],
            'test': [
                [(111, 108, 441, 193)],
                [(45, 49, 499, 162),
                 (2, 69, 437, 207)],
                [(38, 19, 385, 373)],
            ],
        }
    }
    _SIZE_VERIFICATION = {
        'CUB': {
            'val': [
                (560, 419),
                (576, 432),
                (999, 587),
            ],
            'test': [
                (500, 335),
                (500, 347),
                (500, 470),
            ],
        },
        'ILSVRC': {
            'val': [
                (500, 375),
                (500, 375),
                (500, 375),
            ],
            'test': [
                (500, 375),
                (500, 375),
                (500, 375),
            ],
        }
    }

    def _check_box_sizes(self, image_ids, evaluator):
        for image_id in image_ids:
            boxes_in_image = evaluator.original_bboxes[image_id]
            size = evaluator.image_sizes[image_id]
            print(image_id)
            for box in boxes_in_image:
                x0, y0, x1, y1 = box
                width, height = size
                self.assertLessEqual(x0, width - 1)
                self.assertLessEqual(x1, width - 1)
                self.assertGreaterEqual(x0, 0)
                self.assertGreaterEqual(x1, 0)
                self.assertGreaterEqual(x1, x0)
                self.assertLessEqual(y0, height - 1)
                self.assertLessEqual(y1, height - 1)
                self.assertGreaterEqual(y0, 0)
                self.assertGreaterEqual(y1, 0)
                self.assertGreaterEqual(y1, y0)

    def test_box_evaluator_boxes_and_sizes(self):
        for dataset_name in self._DATASET_NAMES:
            for split in self._SPLITS:
                metadata = set_metadata(
                    dataset_name=dataset_name,
                    split=split)
                evaluator = load_evaluator(
                    BoxEvaluator,
                    dataset_name=dataset_name,
                    split=split,
                    cam_curve_interval=self._CAM_CURVE_INTERVAL)
                image_ids = get_image_ids(metadata)

                test_boxes = [evaluator.original_bboxes[image_id]
                              for image_id in image_ids[:3]]
                test_sizes = [evaluator.image_sizes[image_id]
                              for image_id in image_ids[:3]]
                self.assertEqual(
                    test_boxes, self._BOX_VERIFICATION[dataset_name][split])
                self.assertEqual(
                    test_sizes, self._SIZE_VERIFICATION[dataset_name][split])
                self._check_box_sizes(image_ids, evaluator)


class MaskEvaluatorTest(unittest.TestCase):
    _DATASET_NAME = 'OpenImages'
    _SPLIT = 'val'
    _TEST_IMAGE_IDS = (
        'val/0bt_c3/1cd9ac0169ec7df0.jpg',
        'val/0bt_c3/3a29f0b33894d4a6.jpg',
        'val/0bt_c3/3b3b9931325d65c9.jpg',
    )
    _TEST_CONSTANT_GT_IMAGE_IDS = (
        'val/0by6g/490365f63135f3e0.jpg',
        'val/0by6g/b4d953e7105da1d8.jpg',
    )
    _IGNORE_PATHS = {
        'val/0bt_c3/1cd9ac0169ec7df0.jpg':
            'val/0bt_c3/1cd9ac0169ec7df0_ignore.png',
        'val/0bt_c3/3a29f0b33894d4a6.jpg':
            'val/0bt_c3/3a29f0b33894d4a6_ignore.png',
        'val/0bt_c3/3b3b9931325d65c9.jpg':
            'val/0bt_c3/3b3b9931325d65c9_ignore.png',
        'val/0by6g/490365f63135f3e0.jpg':
            'val/0by6g/b4d953e7105da1d8_ignore.png',
        'val/0by6g/b4d953e7105da1d8.jpg':
            'val/0by6g/490365f63135f3e0_ignore.png',
    }
    _MASK_PATHS = {
        'val/0bt_c3/1cd9ac0169ec7df0.jpg':
            ['val/0bt_c3/1cd9ac0169ec7df0_m0bt_c3_6932e993.png'],
        'val/0bt_c3/3a29f0b33894d4a6.jpg':
            ['val/0bt_c3/3a29f0b33894d4a6_m0bt_c3_3581e3ae.png',
             'val/0bt_c3/3a29f0b33894d4a6_m0bt_c3_d2422806.png'],
        'val/0bt_c3/3b3b9931325d65c9.jpg':
            ['val/0bt_c3/3b3b9931325d65c9_m0bt_c3_d4e2e136.png'],
        'val/0by6g/490365f63135f3e0.jpg':
            ['val/0by6g/b4d953e7105da1d8_m0by6g_9f8621ff.png'],
        'val/0by6g/b4d953e7105da1d8.jpg':
            ['val/0by6g/490365f63135f3e0_m0by6g_4f9ded4f.png'],
    }
    _CAM_CURVE_INTERVAL = 0.01
    _MASK_ROOT = 'test_data/test_masks'
    _VALIDATION_SCORES = (0.0, 1.0)

    def _get_perfect_scoremap(self, image_id, ignore_score):
        mask = get_mask(self._MASK_ROOT, self._MASK_PATHS[image_id],
                        self._IGNORE_PATHS[image_id])
        ignore_region = mask == 255
        scoremap = mask.astype(np.float)
        scoremap[ignore_region] = ignore_score
        return scoremap

    def _get_constant_scoremap(self, image_id, value):
        mask = get_mask(self._MASK_ROOT, self._MASK_PATHS[image_id],
                        self._IGNORE_PATHS[image_id])
        scoremap = np.ones_like(mask).astype(np.float)
        return scoremap * value

    def _constant_scoremap_test_template(self, image_id, value):
        evaluator = load_evaluator(
            MaskEvaluator,
            dataset_name=self._DATASET_NAME,
            split=self._SPLIT,
            cam_curve_interval=self._CAM_CURVE_INTERVAL,
            mask_root=self._MASK_ROOT)
        scoremap = self._get_constant_scoremap(image_id, value)
        evaluator.accumulate(scoremap, image_id)
        auc = evaluator.compute()
        return auc

    def test_mask_evaluator_masks_perfect_scoremap(self):
        for image_id in self._TEST_IMAGE_IDS:
            for ignore_score in self._VALIDATION_SCORES:
                evaluator = load_evaluator(
                    MaskEvaluator,
                    dataset_name=self._DATASET_NAME,
                    split=self._SPLIT,
                    cam_curve_interval=self._CAM_CURVE_INTERVAL,
                    mask_root=self._MASK_ROOT)
                scoremap = self._get_perfect_scoremap(
                    image_id, ignore_score=ignore_score)
                evaluator.accumulate(scoremap, image_id)
                auc = evaluator.compute()
                self.assertEqual(auc, 100.0)

    def test_mask_evaluator_masks_zero_scoremap_mask0(self):
        image_id = self._TEST_IMAGE_IDS[0]
        value = 0.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 43031 / (43031 + 7145), delta=0.1)

    def test_mask_evaluator_masks_half_scoremap_mask0(self):
        image_id = self._TEST_IMAGE_IDS[0]
        value = 0.5
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 43031 / (43031 + 7145), delta=0.1)

    def test_mask_evaluator_masks_one_scoremap_mask0(self):
        image_id = self._TEST_IMAGE_IDS[0]
        value = 1.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 43031 / (43031 + 7145), delta=0.1)

    def test_mask_evaluator_masks_zero_scoremap_mask1(self):
        image_id = self._TEST_IMAGE_IDS[1]
        value = 0.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 833 / (833 + 49342), delta=0.1)

    def test_mask_evaluator_masks_half_scoremap_mask1(self):
        image_id = self._TEST_IMAGE_IDS[1]
        value = 0.5
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 833 / (833 + 49342), delta=0.1)

    def test_mask_evaluator_masks_one_scoremap_mask1(self):
        image_id = self._TEST_IMAGE_IDS[1]
        value = 1.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 833 / (833 + 49342), delta=0.1)

    def test_mask_evaluator_masks_zero_scoremap_mask2(self):
        image_id = self._TEST_IMAGE_IDS[2]
        value = 0.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 347 / (347 + 13909), delta=0.1)

    def test_mask_evaluator_masks_half_scoremap_mask2(self):
        image_id = self._TEST_IMAGE_IDS[2]
        value = 0.5
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 347 / (347 + 13909), delta=0.1)

    def test_mask_evaluator_masks_one_scoremap_mask2(self):
        image_id = self._TEST_IMAGE_IDS[2]
        value = 1.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100. * 347 / (347 + 13909), delta=0.1)

    def test_mask_evaluator_masks_zero_scoremap_zero_gt(self):
        image_id = self._TEST_CONSTANT_GT_IMAGE_IDS[0]
        value = 0.0
        self.assertRaises(RuntimeError, self._constant_scoremap_test_template,
                          image_id, value)

    def test_mask_evaluator_masks_one_scoremap_zero_gt(self):
        image_id = self._TEST_CONSTANT_GT_IMAGE_IDS[0]
        value = 1.0
        self.assertRaises(RuntimeError, self._constant_scoremap_test_template,
                          image_id, value)

    def test_mask_evaluator_masks_zero_scoremap_one_gt(self):
        image_id = self._TEST_CONSTANT_GT_IMAGE_IDS[1]
        value = 0.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100.0, delta=0.1)

    def test_mask_evaluator_masks_one_scoremap_one_gt(self):
        image_id = self._TEST_CONSTANT_GT_IMAGE_IDS[1]
        value = 1.0
        auc = self._constant_scoremap_test_template(image_id, value)
        self.assertAlmostEqual(auc, 100.0, delta=0.1)

    def test_mask_evaluator_masks_nan_scoremap(self):
        image_id = self._TEST_CONSTANT_GT_IMAGE_IDS[1]
        value = np.full(1, np.nan)
        self.assertRaises(ValueError, self._constant_scoremap_test_template,
                          image_id, value)


if __name__ == '__main__':
    unittest.main()
