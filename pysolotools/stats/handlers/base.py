from abc import ABC, abstractmethod
from typing import Any


class StatsHandler(ABC):
    """
    Abstract class defining the essential interfaces to process the dataset
    and handle the stats.
    """

    @abstractmethod
    def handle(self, **kwargs: Any) -> dict:
        """
        Returns dictionary where key is stat class name and value are computed stats.

        Returns:
            A dictionary of all stats.

        """
        pass
