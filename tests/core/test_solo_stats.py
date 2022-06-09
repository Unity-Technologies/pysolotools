import numpy as np
import pytest

from pysolotools.consumers import Solo
from pysolotools.core.stats import SoloStats

solo = Solo("tests/data/solo")
stats = solo.stats


def test_stats():
    assert isinstance(solo.stats, SoloStats)


def test_get_categories():
    cats = stats.get_categories()
    expected_cats = {1: "Crate", 2: "Cube", 3: "Box", 4: "Terrain", 5: "Character"}
    assert cats == expected_cats


def test_get_frame_ids():
    frame_ids = stats.get_frame_ids()
    expected_frame_ids = [2]
    assert isinstance(frame_ids, list)
    assert frame_ids == expected_frame_ids


@pytest.mark.parametrize(
    "test_input, expected", [(None, 3), ([1], 2), ([5], 1), ([1, 5], 3)]
)
def test_get_num_bbox(test_input, expected):
    num_bbox = stats.get_num_bbox(cat_ids=test_input)
    assert num_bbox == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [(None, {3: 1}), ([1], {2: 1}), ([5], {1: 1}), ([1, 5], {3: 1})],
)
def test_get_bbox_per_img_dist(test_input, expected):
    bbox_dist = stats.get_bbox_per_img_dist(cat_ids=test_input)
    assert bbox_dist == expected


def test_bbox_heatmap():
    hm = stats.get_bbox_heatmap()
    assert isinstance(hm, np.ndarray)


def test_bbox_size_dist():
    bbox_relative_size = stats.get_bbox_size_dist()
    expected = [0.23593232610221093, 0.23385358667337133, 0.28133136843798184]
    assert isinstance(bbox_relative_size, list)
    assert bbox_relative_size == expected
