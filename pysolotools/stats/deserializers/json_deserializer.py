import json

from pysolotools.clients.file_strategy import FileStrategy, NoOpFileStrategy
from pysolotools.stats.deserializers.base import Deserializer


class JsonDeserializer(Deserializer):
    def __init__(self, file_strategy: FileStrategy = NoOpFileStrategy()):
        self.file_strategy = file_strategy

    def deserialize(self) -> dict:
        """
        Deserialize data from file and map it to dictionary.
        Args:

        Returns:
            Deserialized object.

        """

        return json.loads(self.file_strategy.read())
