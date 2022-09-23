from abc import ABC, abstractmethod


class Deserializer(ABC):
    """
    Abstract class defining the essential interfaces to deserialize file to object.
    """

    @abstractmethod
    def deserialize(self) -> object:
        """
        Deserialize data from file and map it to object.
        Args:

        Returns:
            Deserialized object.

        """
        pass
