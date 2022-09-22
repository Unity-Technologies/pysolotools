from abc import ABC, abstractmethod
from typing import Any


class DeSerializer(ABC):
    """
    Abstract class defining the essential interfaces to deserialize file to object.
    """

    @abstractmethod
    def deserialize(self, src_path: str = None, **kwargs: Any) -> object:
        """
        Deserialize data from file and map it to object.
        Args:
            src_path (str): source file path.

        Returns:
            Deserialized object.

        """
        pass
