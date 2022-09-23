import json
import logging

import numpy as np

from pysolotools.clients.file_strategy import FileStrategy, NoOpFileStrategy
from pysolotools.stats.serializers.base import Serializer

logger = logging.getLogger(__name__)


class JsonSerializer(Serializer):
    def __init__(self, file_strategy: FileStrategy = NoOpFileStrategy()):
        self.file_strategy = file_strategy
        logger.info(f"Using file_strategy: {type(self.file_strategy).__name__}")

    def serialize(self, data: object = None, **kwargs):
        """
        Serialize object to json file.

        Args:
            data (object): data that we want to serialize.

        """
        compatible_data = _pre_process(data)
        contents = json.dumps(compatible_data)
        self.file_strategy.write(contents=contents)


def _pre_process(data: object):
    """
    pre-process the data to make json data type

    """
    compatible_data = data.copy() if data else {}
    for k, v in compatible_data.items():
        if isinstance(v, dict):
            compatible_data[k] = _pre_process(v)
        if isinstance(v, np.ndarray):
            compatible_data[k] = v.tolist()

    return compatible_data
