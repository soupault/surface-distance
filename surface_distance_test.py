# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple tests for surface metric computations."""

import numpy as np
import surface_distance

from numpy.testing import assert_almost_equal


def assert_metrics(surface_distances, mask_gt, mask_pred,
                   expected_average_surface_distance,
                   expected_hausdorff_100,
                   expected_hausdorff_95,
                   expected_surface_overlap_at_1mm,
                   expected_surface_dice_at_1mm,
                   expected_volumetric_dice,
                   decimal=3):
    actual_average_surface_distance = (
        surface_distance.compute_average_surface_distance(surface_distances))

    for i in range(2):
      assert_almost_equal(
          expected_average_surface_distance[i],
          actual_average_surface_distance[i],
          decimal=decimal)

    assert_almost_equal(
        expected_hausdorff_100,
        surface_distance.compute_robust_hausdorff(surface_distances, 100),
        decimal=decimal)

    assert_almost_equal(
        expected_hausdorff_95,
        surface_distance.compute_robust_hausdorff(surface_distances, 95),
        decimal=decimal)

    actual_surface_overlap_at_1mm = (
        surface_distance.compute_surface_overlap_at_tolerance(
            surface_distances, 1))
    for i in range(2):
      assert_almost_equal(
          expected_surface_overlap_at_1mm[i],
          actual_surface_overlap_at_1mm[i],
          decimal=decimal)

    assert_almost_equal(
        expected_surface_dice_at_1mm,
        surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, 1),
        decimal=decimal)

    assert_almost_equal(
        expected_volumetric_dice,
        surface_distance.compute_dice_coefficient(mask_gt, mask_pred),
        decimal=decimal)


def testSinglePixels2mmAway():
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    mask_gt[50, 60, 70] = 1
    mask_pred[50, 60, 72] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    assert_metrics(surface_distances, mask_gt, mask_pred,
                         expected_average_surface_distance=(1.5, 1.5),
                         expected_hausdorff_100=2.0,
                         expected_hausdorff_95=2.0,
                         expected_surface_overlap_at_1mm=(0.5, 0.5),
                         expected_surface_dice_at_1mm=0.5,
                         expected_volumetric_dice=0.0)


def testTwoCubes():
    mask_gt = np.zeros((100, 100, 100), np.uint8)
    mask_pred = np.zeros((100, 100, 100), np.uint8)
    mask_gt[0:50, :, :] = 1
    mask_pred[0:51, :, :] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(2, 1, 1))
    assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(0.322, 0.339),
        expected_hausdorff_100=2.0,
        expected_hausdorff_95=2.0,
        expected_surface_overlap_at_1mm=(0.842, 0.830),
        expected_surface_dice_at_1mm=0.836,
        expected_volumetric_dice=0.990)


def testEmptyPredictionMask():
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    mask_gt[50, 60, 70] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(np.inf, np.nan),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(0.0, np.nan),
        expected_surface_dice_at_1mm=0.0,
        expected_volumetric_dice=0.0)


def testEmptyGroundTruthMask():
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    mask_pred[50, 60, 72] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(np.nan, np.inf),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(np.nan, 0.0),
        expected_surface_dice_at_1mm=0.0,
        expected_volumetric_dice=0.0)


def testEmptyBothMasks():
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(np.nan, np.nan),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(np.nan, np.nan),
        expected_surface_dice_at_1mm=np.nan,
        expected_volumetric_dice=np.nan)


if __name__ == "__main__":
    np.testing.run_module_suite()
