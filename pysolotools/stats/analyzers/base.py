from abc import ABC, abstractmethod
from typing import Any

from pysolotools.core.models import Frame


class StatsAnalyzer(ABC):
    """
    Abstract class defining the essential interfaces to compute the stats.
    """

    @abstractmethod
    def analyze(self, frame: Frame = None, **kwargs: Any) -> Any:
        """
        Returns computed stats values.
        Args:
            frame (object): frame object.

        Returns:
            computed stats values.

        """
        pass

    @abstractmethod
    def merge(self, frame_result: Any, **kwargs: Any):
        """
        Merge computed stats values.
        Args:
            frame_result (object):  result of one frame.

        """
        pass

    @abstractmethod
    def get_result(self) -> Any:
        """
        Returns results stored in the instance at any given point
        """
        pass
