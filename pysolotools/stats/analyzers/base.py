import logging
from abc import ABC, abstractmethod
from typing import Any, List

from pysolotools.core.models import Frame

logger = logging.getLogger(__name__)


class StatsAnalyzer(ABC):
    """
    Abstract class defining the essential interfaces to compute the stats.
    """

    @abstractmethod
    def analyze(self, frame: Frame = None, cat_ids: List = None, **kwargs: Any) -> Any:
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
    def merge(self, agg_result: Any, frame_result: Any, **kwargs: Any) -> Any:
        """
        Merge computed stats values.
        Args:
            agg_result (object): aggregated results.
            frame_result (object):  result of one frame.

        Returns:
            aggregated stats values.

        """
        pass
