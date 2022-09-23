from abc import ABC, abstractmethod
from typing import Any


class Serializer(ABC):
    """
    Abstract class defining the essential interfaces to serializing the object.
    """

    @abstractmethod
    def serialize(self, data: object = None, **kwargs: Any):
        """
        Serialize object to given path.

        Args:
            data (object): data that we want to serialize.

        """
        pass
