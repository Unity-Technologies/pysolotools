from typing import Any, Union
from unittest.mock import Mock, create_autospec

import pytest

from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.base import StatsAnalyzer
from pysolotools.stats.handler import StatsHandler
from pysolotools.stats.serializers.base import Serializer


class FakeAnalyzer1(StatsAnalyzer):
    def __init__(self):
        self._res = []

    def analyze(self, frame: object = None, **kwargs: Any) -> object:
        return 3

    def merge(self, frame_result: Any, **kwargs: Any) -> Any:
        self._res.append(frame_result)

    def get_result(self) -> Any:
        return self._res


class FakeAnalyzer2(StatsAnalyzer):
    def __init__(self):
        self._res = {"count": 0}

    def analyze(self, frame: object = None, **kwargs: Any) -> object:
        return {"count": 3}

    def merge(self, frame_result: Any, **kwargs: Any) -> Any:
        self._res["count"] += frame_result["count"]

    def get_result(self) -> Any:
        return self._res


@pytest.mark.parametrize(
    "description, mock_serializer, mock_analyzers, mock_frames, expected_result",
    [
        (
            "1 analyzer, 1 frame, no serializer",
            None,
            [FakeAnalyzer1()],
            [Mock()],
            {"FakeAnalyzer1": [3]},
        ),
        (
            "1 analyzer, 1 frame, with serializer",
            create_autospec(spec=Serializer, spec_set=True),
            [FakeAnalyzer1()],
            [Mock()],
            {"FakeAnalyzer1": [3]},
        ),
        (
            "1 analyzer, 2 frames, no serializer",
            None,
            [FakeAnalyzer1()],
            [Mock(), Mock()],
            {"FakeAnalyzer1": [3, 3]},
        ),
        (
            "2 analyzer, 1  frame, no serializer",
            None,
            [FakeAnalyzer1(), FakeAnalyzer2()],
            [Mock(), Mock()],
            {
                "FakeAnalyzer1": [3, 3],
                "FakeAnalyzer2": {"count": 6},
            },
        ),
        ("0 analyzer, 0 frame, no serializer", None, [], [], {}),
        (
            "1 analyzer, 0 frame, no serializer",
            None,
            [FakeAnalyzer1()],
            [],
            {"FakeAnalyzer1": []},
        ),
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
