import json
from unittest.mock import create_autospec, patch

import numpy as np
import pytest

from pysolotools.clients.file_strategy import FileStrategy
from pysolotools.stats.serializers.json_serializer import JsonSerializer, _pre_process


class TestJsonSerializer:
    @patch("pysolotools.stats.serializers.json_serializer._pre_process", autospec=True)
    def test_serialize(self, mock_pre_process):
        data = {"foo": "bar"}
        expected_data = json.dumps(data)
        mock_pre_process.return_value = data
        mock_file_strategy = create_autospec(spec=FileStrategy)
        subject = JsonSerializer(file_strategy=mock_file_strategy)
        subject.serialize(data=data)
        mock_pre_process.assert_called_once_with(data)
        mock_file_strategy.write.assert_called_once_with(contents=expected_data)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {"foo": "bar", "some_numpy": np.array([[1, 0], [0, 1]], np.int32)},
            {"foo": "bar", "some_numpy": [[1, 0], [0, 1]]},
        ),
        (
            {
                "foo": "bar",
                "some_object": {"some_numpy": np.array([[1, 0], [0, 1]], np.int32)},
            },
            {"foo": "bar", "some_object": {"some_numpy": [[1, 0], [0, 1]]}},
        ),
        ({}, {}),
        (None, {}),
    ],
)
def test_pre_process(test_input, expected):
    assert _pre_process(data=test_input) == expected
