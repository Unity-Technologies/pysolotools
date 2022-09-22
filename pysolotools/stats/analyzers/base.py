from abc import ABC, abstractmethod
from typing import Any


class StatsAnalyzer(ABC):
    """
    Abstract class defining the essential interfaces to compute the stats.
    """

    @abstractmethod
    def analyze(
        self, frame: object = None, cat_ids: list = None, **kwargs: Any
    ) -> object:
        """
        Returns computed stats values.
        Args:
            frame (object): frame object.
            cat_ids(list): list of categories

        Returns:
            computed stats values.

        """
        pass

    @abstractmethod
    def merge(
        self, results: object = None, result: object = None, **kwargs: Any
    ) -> object:
        """
        Merge computed stats values.
        Args:
            results (object): aggregated results.
            result (object):  result of one frame.

        Returns:
            aggregated stats values.

        """
        pass
