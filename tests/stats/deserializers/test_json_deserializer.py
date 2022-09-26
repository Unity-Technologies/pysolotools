import json
from unittest.mock import create_autospec

from pysolotools.clients.file_strategy import FileStrategy
from pysolotools.stats.deserializers.json_deserializer import JsonDeserializer


class TestJsonDeserializer:
    def test_deserialize(self):
        data = {"foo": "bar"}
        mock_file_strategy = create_autospec(spec=FileStrategy)
        mock_file_strategy.read.return_value = json.dumps(data)
        subject = JsonDeserializer(file_strategy=mock_file_strategy)
        assert subject.deserialize() == data
