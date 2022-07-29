import os
from pathlib import Path

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pysolotools.consumers.solo_stats import SoloStats, TOTAL_TOKEN


class TestConsumerSoloStats:

    @pytest.fixture(scope='class')
    def setup(self):
        stats_path = os.path.join(Path(__file__).parents[1], 'data', 'iterative_stats')
        solo_stats = SoloStats(solo_path=stats_path)
        yield solo_stats

    def test_total_object_count(self, setup):
        assert setup.total_object_count == 25

    def test_category_labels(self, setup):
        assert setup.category_labels == ['Cube', 'Box', 'Crate', 'Terrain']

    def test_category_ids(self, setup):
        assert setup.category_ids == [2, 3, 1, 4]

    def test_frame_ids(self, setup):
        assert setup.frame_ids == [2, 3, 4, 5, 6]

    @pytest.mark.parametrize(
        "test_input, expected", [
            (None, 25),  # this case seems weird
            ([], 25),
            ([TOTAL_TOKEN], 25),
            (['Box', TOTAL_TOKEN], 25),
            (['Cube'], 5),
        ]
    )
    def test_get_object_count(self, setup, test_input, expected):
        assert setup.get_object_count(labels=test_input) == expected

    @patch.object(SoloStats, 'get_object_count')
    def test_get_object_count(self, mock_get_object_count, setup):
        assert setup.get_object_count_by_id(ids=[1, 2]) == mock_get_object_count.return_value
        mock_get_object_count.assert_called_once_with(['Cube', 'Crate'])

    @patch.object(SoloStats, 'get_bbox_per_img_dist_by_labels')
    def test_get_bbox_per_img_dist_by_ids(self, mock_get_bbox_per_img_dist_by_ids, setup):
        assert setup.get_bbox_per_img_dist_by_ids(category_ids=[1, 2]) == mock_get_bbox_per_img_dist_by_ids.return_value
        mock_get_bbox_per_img_dist_by_ids.assert_called_once_with(['Cube', 'Crate'])

    @pytest.mark.parametrize(
        "test_input, expected", [
            (['Cube', 'Crate'], {'2': 3, '3': 3, '4': 3, '5': 3, '6': 3}),
            ([TOTAL_TOKEN], {'2': 5, '3': 5, '4': 5, '5': 5, '6': 5}),
            (None, {'2': 5, '3': 5, '4': 5, '5': 5, '6': 5}),
        ]
    )
    def test_get_bbox_per_img_dist_by_labels(self, setup, test_input, expected):
        assert setup.get_bbox_per_img_dist_by_labels(category_ids=test_input) == expected

    @pytest.mark.parametrize(
        "test_input, expected", [
            (['Cube'], []),
        ]
    )
    def test_get_bbox_heatmap_by_labels(self, setup, test_input, expected):
        np.set_printoptions(threshold=np.inf)
        print(setup.get_bbox_heatmap_by_labels(category_ids=test_input))
        # assert setup.get_bbox_heatmap_by_labels(category_ids=test_input) == expected

