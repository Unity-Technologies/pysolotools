from unittest.mock import Mock, patch

import numpy as np

from pysolotools.core.models import BoundingBox2DAnnotation, BoundingBox2DLabel
from pysolotools.stats.analyzers.bbox_analyzer import (
    BBoxCountStats,
    BBoxCountStatsAnalyzer,
    BBoxHeatMapStatsAnalyzer,
    BBoxSizeStatsAnalyzer,
)


class TestBBoxCountStatsAnalyzer:
    def test_analyze(self, solo_instance):
        analyzer = BBoxCountStatsAnalyzer(solo_instance)
        frame = next(solo_instance.frames())
        res = analyzer.analyze(frame)
        assert len(res[1].keys()) == 2
        assert res[1][1] == 2
        assert res[1][5] == 1

    def test_merge(self, solo_instance):
        analyzer = BBoxCountStatsAnalyzer(solo_instance)
        analyzer.merge((2, {1: 2, 5: 1}))
        analyzer.merge((3, {1: 3, 5: 2, 4: 1}))
        assert analyzer.get_result().get_total_count() == 9


class TestBBoxCountStats:
    def test_get_ids(self, solo_instance):
        stats = BBoxCountStats(solo_instance)
        assert len(stats.get_ids()) == 5

    def test_get_labels(self, solo_instance):
        stats = BBoxCountStats(solo_instance)
        assert len(stats.get_labels()) == 5
        assert "Crate" in stats.get_labels()

    def test_add_counts(self, solo_instance):
        stats = BBoxCountStats(solo_instance)
        stats.add_counts((2, {1: 2, 5: 1}))
        stats.add_counts((3, {1: 3, 5: 2, 4: 1}))
        assert stats.get_total_count() == 9

    def test_get_count(self, solo_instance):
        stats = BBoxCountStats(solo_instance)
        stats.add_counts((2, {1: 2, 5: 1}))
        stats.add_counts((3, {1: 3, 5: 2, 4: 1}))
        assert stats.get_count([1]) == 5
        assert stats.get_count([5]) == 3
        assert stats.get_count([4]) == 1
        assert stats.get_count([99]) == 0
        assert stats.get_count([1, 5]) == 8

    def test_get_count_by_label(self, solo_instance):
        stats = BBoxCountStats(solo_instance)
        stats.add_counts((2, {1: 2, 5: 1}))
        stats.add_counts((3, {1: 3, 5: 2, 4: 1}))
        assert stats.get_count_by_label(["Crate"]) == 5
        assert stats.get_count_by_label(["Character"]) == 3
        assert stats.get_count_by_label(["Terrain"]) == 1
        assert stats.get_count_by_label(["DNE"]) == 0
        assert stats.get_count_by_label(["Crate", "Character"]) == 8

    def test_get_count_per_frame(self, solo_instance):
        stats = BBoxCountStats(solo_instance)
        stats.add_counts((2, {1: 2, 5: 1}))
        stats.add_counts((3, {1: 3, 5: 2, 4: 1}))
        assert stats.get_count_per_frame([2])[2] == 3
        assert stats.get_count_per_frame([3])[3] == 6
        assert stats.get_count_per_frame([2], [1])[2] == 2
        assert stats.get_count_per_frame([3], [4])[3] == 1

    def test_get_count_per_frame_by_label(self, solo_instance):
        stats = BBoxCountStats(solo_instance)
        stats.add_counts((2, {1: 2, 5: 1}))
        stats.add_counts((3, {1: 3, 5: 2, 4: 1}))
        assert stats.get_count_per_frame_by_label([2], ["Crate"])[2] == 2
        assert stats.get_count_per_frame_by_label([3], ["Terrain"])[3] == 1


class TestBBoxStatsAnalyzer:
    def test_analyze(self, solo_instance):
        bbox_size_analyzer = BBoxSizeStatsAnalyzer()
        frame = next(solo_instance.frames())
        res = bbox_size_analyzer.analyze(frame)
        assert len(res) == 3

    def test_merge(self):
        bbox_size_analyzer = BBoxSizeStatsAnalyzer()
        bbox_size_analyzer.merge([1, 2, 3])
        bbox_size_analyzer.merge([4, 5, 6])
        assert bbox_size_analyzer.get_result() == [1, 2, 3, 4, 5, 6]

    def test_analyze_with_cat_id(self, solo_instance):
        bbox_size_analyzer = BBoxSizeStatsAnalyzer(cat_ids=[5])
        for frame in solo_instance.frames():
            res = bbox_size_analyzer.analyze(frame)
            bbox_size_analyzer.merge(res)
        assert bbox_size_analyzer.get_result() == [
            0.23385358667337133,
            0.23385358667337133,
        ]


class TestBBoxHeatMapStatsAnalyzer:
    def test_analyze(self, solo_instance):
        bbox_hm_analyzer = BBoxHeatMapStatsAnalyzer()
        frame = next(solo_instance.frames())
        res = bbox_hm_analyzer.analyze(frame)
        assert np.sum(res) > 0.0

    def test_merge(self):
        res_1 = np.array([[1, 2, 3], [4, 5, 6]])
        res_2 = np.array([[1, 2, 3], [4, 5, 6]])

        expected_res = np.array([[2, 4, 6], [8, 10, 12]])

        bbox_hm_analyzer = BBoxHeatMapStatsAnalyzer()
        bbox_hm_analyzer.merge(res_1)
        bbox_hm_analyzer.merge(res_2)

        assert (
            np.testing.assert_array_equal(bbox_hm_analyzer.get_result(), expected_res)
            is None
        )

    @patch("pysolotools.stats.analyzers.bbox_analyzer._frame_bbox_dim", autospec=True)
    def test_analyze_with_cat_id(self, mock_frame_bbox_dim):
        bbox_2d_label_1 = {
            "instanceId": 3,
            "labelId": 1,
            "labelName": "A",
            "origin": [0.0, 0.0],
            "dimension": [1, 2],
        }
        bbox_2d_label_2 = {
            "instanceId": 3,
            "labelId": 2,
            "labelName": "B",
            "origin": [0.0, 0.0],
            "dimension": [2, 2],
        }
        bb = BoundingBox2DAnnotation(
            type="abc",
            id="bbox",
            sensorId="cam",
            description="xyz",
            values=[
                BoundingBox2DLabel.from_dict(bbox_2d_label_1),
                BoundingBox2DLabel.from_dict(bbox_2d_label_2),
            ],
            extra_data={},
        )
        mock_frame_bbox_dim.return_value = [2, 2], [bb]
        bbox_hm_analyzer = BBoxHeatMapStatsAnalyzer(cat_ids=[2])
        res = bbox_hm_analyzer.analyze(Mock())
        assert res.tolist() == [[[1.0], [1.0]], [[1.0], [1.0]]]
