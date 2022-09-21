from typing import Any, Union
from unittest.mock import Mock, create_autospec

import pytest

from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.base import StatsAnalyzer
from pysolotools.stats.handlers.handler import StatsHandler
from pysolotools.stats.serializers.base import Serializer


class FakeAnalyzer1(StatsAnalyzer):
    def analyze(
        self, frame: object = None, cat_ids: list = None, **kwargs: Any
    ) -> object:
        return f"analyze-return-1-{frame}"

    def merge(
        self, results: object = None, result: object = None, **kwargs: Any
    ) -> object:
        return f"merge-return-1-{result}"


class FakeAnalyzer2(StatsAnalyzer):
    def analyze(
        self, frame: object = None, cat_ids: list = None, **kwargs: Any
    ) -> object:
        return f"analyze-return-2-{frame}"

    def merge(
        self, results: object = None, result: object = None, **kwargs: Any
    ) -> object:
        return f"merge-return-2-{result}"


@pytest.mark.parametrize(
    "description, mock_serializer, mock_analyzers, mock_frames, expected_result",
    [
        (
            "1 analyzer, 1 frame, no serializer",
            None,
            [FakeAnalyzer1()],
            ["some-frame-data"],
            {"FakeAnalyzer1": "analyze-return-1-some-frame-data"},
        ),
        (
            "1 analyzer, 1 frame, with serializer",
            create_autospec(spec=Serializer, spec_set=True),
            [FakeAnalyzer1()],
            ["some-frame-data"],
            {"FakeAnalyzer1": "analyze-return-1-some-frame-data"},
        ),
        (
            "1 analyzer, 2 frames, no serializer",
            None,
            [FakeAnalyzer1()],
            ["some-frame-data", "some-other-frame-data"],
            {"FakeAnalyzer1": "merge-return-1-analyze-return-1-some-other-frame-data"},
        ),
        (
            "2 analyzer, 1  frame, no serializer",
            None,
            [FakeAnalyzer1(), FakeAnalyzer2()],
            ["some-frame-data", "some-other-frame-data"],
            {
                "FakeAnalyzer1": "merge-return-1-analyze-return-1-some-other-frame-data",
                "FakeAnalyzer2": "merge-return-2-analyze-return-2-some-other-frame-data",
            },
        ),
        ("0 analyzer, 0 frame, no serializer", None, [], [], {}),
        ("1 analyzer, 0 frame, no serializer", None, [FakeAnalyzer1()], [], {}),
    ],
)
def test_stats_handler(
    description,
    mock_serializer: Union[Mock, None],
    mock_analyzers,
    mock_frames,
    expected_result,
):
    mock_solo = create_autospec(spec=Solo)
    mock_solo.frames.return_value = mock_frames
    handler = StatsHandler(solo=mock_solo)
    result = handler.handle(analyzers=mock_analyzers, serializer=mock_serializer)

    assert result == expected_result
    if mock_serializer:
        mock_serializer.serialize.assert_called_once_with(expected_result)
