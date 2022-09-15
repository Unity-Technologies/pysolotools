import json
from pathlib import Path

import numpy as np

from pysolotools.stats.serializers.base import Serializer


class JsonSerializer(Serializer):
    def __init__(self, dest_dir: str = "", file_name: str = ""):
        self.dest_dir = dest_dir
        self.file_name = file_name

    def serialize(self, data: object = None):
        """
        Serialize object to json file.

        Args:
            data (object): data that we want to serialize.

        """
        dest_dir = Path(self.dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_file = dest_dir / f"{self.file_name}.json"
        compatible_data = _pre_process(data)
        print("writing data to file", output_file)
        with open(output_file, "w") as out:
            json.dump(compatible_data, out)


def _pre_process(data):
    """
    pre-process the data to make json data type

    """
    compatible_data = data.copy()
    for k, v in compatible_data.items():
        if isinstance(v, np.ndarray):
            compatible_data[k] = v.tolist()

    return compatible_data
