from typing import Any

from pysolotools.consumers import Solo

from pysolotools.stats.handlers.base import StatsHandler
from pysolotools.stats.serializers.base import Serializer


class BBoxHandler(StatsHandler):
    """
    Compute Bounding box stats and returns dictionary where key is stat class name and value are computed stats.

    Args:
        solo (SOLO): data that we want to serialize.
        analyzers (list): list of analyzers.
        serializers (Serializer): serializer object.

    """

    def __init__(
        self,
        solo: Solo = None,
        analyzers: list = [],
        serializer: Serializer = None,
        **kwargs: Any
    ):
        self.solo = solo
        self.analyzers = analyzers
        self.serializer = serializer

    def handle(self, **kwargs: Any) -> dict:
        pass
