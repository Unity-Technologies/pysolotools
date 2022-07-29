import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pysolotools.consumers.solo_stats import SoloStats, TOTAL_TOKEN, BBOX_HEATMAP_NAME, STATS_PATH_NAME


class TestConsumerSoloStats:

    @pytest.fixture(scope='class')
    def setup(self):
        stats_path = os.path.join(Path(__file__).parents[1], 'data', 'iterative_stats')
        with patch('pysolotools.consumers.solo_stats.np', autospec=True) as mock_np:
            mock_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
            mock_np.load.return_value = {'arr_0': np.array(mock_list)}
            solo_stats = SoloStats(solo_path=stats_path)
            yield solo_stats, mock_np

    def test_constructor(self, setup):
        _, mock_np = setup
        mock_np.load.assert_called_once_with(
            os.path.join(Path(__file__).parents[1], 'data', 'iterative_stats', STATS_PATH_NAME, BBOX_HEATMAP_NAME),
        )

    def test_total_object_count(self, setup):
        solo_stats, _ = setup
        assert solo_stats.total_object_count == 25

    def test_category_labels(self, setup):
        solo_stats, _ = setup
        assert solo_stats.category_labels == ['Cube', 'Box', 'Crate', 'Terrain']

    def test_category_ids(self, setup):
        solo_stats, _ = setup
        assert solo_stats.category_ids == [2, 3, 1, 4]

    def test_frame_ids(self, setup):
        solo_stats, _ = setup
        assert solo_stats.frame_ids == [2, 3, 4, 5, 6]

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
        solo_stats, _ = setup
        assert solo_stats.get_object_count(labels=test_input) == expected

    @patch.object(SoloStats, 'get_object_count')
    def test_get_object_count(self, mock_get_object_count, setup):
        solo_stats, _ = setup
        assert solo_stats.get_object_count_by_id(ids=[1, 2]) == mock_get_object_count.return_value
        mock_get_object_count.assert_called_once_with(['Cube', 'Crate'])

    @patch.object(SoloStats, 'get_bbox_per_img_dist_by_labels')
    def test_get_bbox_per_img_dist_by_ids(self, mock_get_bbox_per_img_dist_by_ids, setup):
        solo_stats, _ = setup
        assert solo_stats.get_bbox_per_img_dist_by_ids(
            category_ids=[1, 2]) == mock_get_bbox_per_img_dist_by_ids.return_value
        mock_get_bbox_per_img_dist_by_ids.assert_called_once_with(['Cube', 'Crate'])

    @pytest.mark.parametrize(
        "test_input, expected", [
            (['Cube', 'Crate'], {'2': 3, '3': 3, '4': 3, '5': 3, '6': 3}),
            ([TOTAL_TOKEN], {'2': 5, '3': 5, '4': 5, '5': 5, '6': 5}),
            (None, {'2': 5, '3': 5, '4': 5, '5': 5, '6': 5}),
        ]
    )
    def test_get_bbox_per_img_dist_by_labels(self, setup, test_input, expected):
        solo_stats, _ = setup
        assert solo_stats.get_bbox_per_img_dist_by_labels(category_ids=test_input) == expected

    @pytest.mark.parametrize(
        "test_input, expected", [
            (['Cube'], np.array([[1, 3], [5, 7]])),
            (['Cube', 'Box'], np.array([[3, 7], [11, 15]])),
        ]
    )
    def test_get_bbox_heatmap_by_labels(self, setup, test_input, expected):
        solo_stats, mock_np = setup
        assert_array_equal(
            x=solo_stats.get_bbox_heatmap_by_labels(category_ids=test_input),
            y=expected
        )

    @pytest.mark.parametrize(
        "test_input, expected", [
            (['Cube'], [[0.28235892495191295, 0.28506098579953026, 0.2895421133732823, 0.2921944629295132, 0.29575446425202107]]),
            (None, [0.28235892495191295, 0.4712640913012575, 0.4798918877813349, 0.05284854539152426, 0.9287087810503355, 0.28506098579953026, 0.47495887979908324, 0.47975620436321614, 0.05284854539152426, 0.9287087810503355, 0.2895421133732823, 0.47510278493185026, 0.4768399124269416, 0.05284854539152426, 0.9287087810503355, 0.2921944629295132, 0.4760952285695233, 0.47469151166836765, 0.053643304956598886, 0.9287087810503355, 0.29575446425202107, 0.47707196129095664, 0.4784006558314903, 0.053643304956598886, 0.9287087810503355]),
        ]
    )
    def test_get_bbox_size_dist_by_labels(self, setup, test_input, expected):
        solo_stats, _ = setup
        assert solo_stats.get_bbox_size_dist_by_labels(category_labels=test_input) == expected
