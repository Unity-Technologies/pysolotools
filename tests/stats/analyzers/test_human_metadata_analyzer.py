from unittest.mock import Mock, create_autospec, patch

import pytest

from pysolotools.core import Frame
from pysolotools.core.models import (
    HumanMetadataAnnotation,
    HumanMetadataLabel,
    NormalAnnotation,
)
from pysolotools.stats.analyzers.human_metadata_analyzer import (
    AggregateObject,
    HumanMetadataAnnotationAggregate,
    HumanMetadataAnnotationAnalyzer,
)


class TestHumanMetadataAnnotation:
    @patch(
        "pysolotools.stats.analyzers.human_metadata_analyzer.HumanMetadataAnnotationAggregate",
        autospec=True,
    )
    def test_constructor(self, mock_annotation):
        HumanMetadataAnnotationAnalyzer()
        mock_annotation.assert_called_once()

    @patch.object(HumanMetadataAnnotationAnalyzer, "_filter", autospec=True)
    def test_analyze(self, mock_filter):
        mock_human_metadata_label = create_autospec(HumanMetadataLabel)
        mock_human_metadata_label.age = "some-age"
        mock_human_metadata_label.sex = "some-sex"
        mock_human_metadata_label.ethnicity = "some-ethnicity"
        mock_human_metadata_annotation = create_autospec(spec=HumanMetadataAnnotation)
        mock_human_metadata_annotation.metadata = [mock_human_metadata_label]
        mock_filter.return_value = [mock_human_metadata_annotation]

        subject = HumanMetadataAnnotationAnalyzer()
        mock_frame = create_autospec(spec=Frame)
        result = subject.analyze(frame=mock_frame)

        mock_filter.assert_called_once_with(frame=mock_frame)
        assert result == [
            AggregateObject(key="human"),
            AggregateObject(key="some-age"),
            AggregateObject(key="some-sex"),
            AggregateObject(key="some-ethnicity"),
        ]

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([AggregateObject(key="foo")], AggregateObject(key="foo", count=1)),
            (
                [AggregateObject(key="foo", value=3)],
                AggregateObject(key="foo", sum=3, count=1),
            ),
            (
                [AggregateObject(key="foo", count=2)],
                AggregateObject(key="foo", count=3),
            ),
        ],
    )
    def test_merge(self, test_input, expected):
        analyzer = HumanMetadataAnnotationAnalyzer()
        analyzer.merge(frame_result=test_input)
        assert analyzer._res.statistics.get("foo") == expected

    def test_filter(self):
        mock_frame = create_autospec(spec=Frame)
        mock_human_metadata = create_autospec(spec=HumanMetadataAnnotation)
        mock_other_annotation = create_autospec(spec=NormalAnnotation)
        mock_capture = Mock()
        mock_capture.annotations = [mock_human_metadata, mock_other_annotation]
        mock_frame.captures = [mock_capture]
        analyzer = HumanMetadataAnnotationAnalyzer()
        assert analyzer._filter(frame=mock_frame) == [mock_human_metadata]

    @patch(
        "pysolotools.stats.analyzers.human_metadata_analyzer.HumanMetadataAnnotationAggregate",
        autospec=True,
    )
    def test_get_result(self, mock_annotation):
        analyzer = HumanMetadataAnnotationAnalyzer()
        result = analyzer.get_result()
        assert result == mock_annotation.return_value.to_dict.return_value


class TestAggregateObject:
    @pytest.mark.parametrize(
        "test_input, value, expected",
        [
            (
                AggregateObject(key="bar"),
                "foo",
                AggregateObject(key="bar", sum=None, count=1),
            ),
            (
                AggregateObject(key="bar", count=3),
                "foo",
                AggregateObject(key="bar", sum=None, count=4),
            ),
            (AggregateObject(key="bar"), 1, AggregateObject(key="bar", sum=1, count=1)),
            (
                AggregateObject(key="bar", sum=10, count=2),
                3,
                AggregateObject(key="bar", sum=13, count=3),
            ),
            (
                AggregateObject(key="bar"),
                1.5,
                AggregateObject(key="bar", sum=1.5, count=1),
            ),
            (
                AggregateObject(key="bar", sum=10.5, count=2),
                3.5,
                AggregateObject(key="bar", sum=14, count=3),
            ),
        ],
    )
    def test_add(self, test_input, value, expected):
        result = test_input.add(value=value)
        assert result == expected


class TestHumanMetadataAnnotationAggregate:
    def test_to_dict(self):
        test_input = {"foo": AggregateObject(key="bar")}
        result = HumanMetadataAnnotationAggregate(statistics=test_input)
        assert result.to_dict() == {
            "foo": {"count": 0, "key": "bar", "sum": None, "value": None}
        }
