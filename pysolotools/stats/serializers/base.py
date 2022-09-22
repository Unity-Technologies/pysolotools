from abc import ABC, abstractmethod
from typing import Any


class Serializer(ABC):
    """
    Abstract class defining the essential interfaces to serializing the object.
    """

    def __init__(self, dest_dir: str = "", file_name: str = "", **kwargs: Any):
        self.dest_dir = dest_dir
        self.file_name = file_name

    @abstractmethod
    def serialize(self, data: object = None, **kwargs: Any):
        """
        Serialize object to given path.

        Args:
            data (object): data that we want to serialize.

        """
        pass
