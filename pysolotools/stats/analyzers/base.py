from abc import ABC, abstractmethod
from typing import Any


class StatsAnalyzer(ABC):
    """
    Abstract class defining the essential interfaces to compute the stats.
    """

    @abstractmethod
    def analyze(self, frame: object = None, **kwargs: Any) -> object:
        """
        Returns computed stats values.
        Args:
            frame (object): frame object.

        Returns:
            computed stats values.

        """
        pass
