import json

from pysolotools.stats.deserializers.base import DeSerializer


class JsonDeSerializer(DeSerializer):
    def deserialize(self, src_path: str) -> dict:
        """
        Deserialize data from file and map it to dictionary.
        Args:
            src_path (str): source file path.

        Returns:
            Deserialized object.

        """

        with open(src_path, "r") as file:
            res = json.load(file)
        return res
