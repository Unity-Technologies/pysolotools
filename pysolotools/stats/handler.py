from typing import Dict, List

from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.base import StatsAnalyzer
from pysolotools.stats.serializers.base import Serializer


class StatsHandler:
    def __init__(self, solo: Solo):
        self.solo = solo

    def handle(
        self,
        analyzers: List[StatsAnalyzer],
        serializer: Serializer = None,
    ) -> Dict:
        """
        Handle stats computation and returns dictionary where key is stat class name and value are computed stats.

        Args:
            analyzers (list): list of analyzers.
            serializer (Serializer): serializer object.

        """

        res = {}
        for i, frame in enumerate(self.solo.frames()):
            for stats_analyzer in analyzers:
                frame_res = stats_analyzer.analyze(frame)
                stats_analyzer.merge(frame_res)

        for stats_analyzer in analyzers:
            class_name = stats_analyzer.__class__.__name__
            res[class_name] = stats_analyzer.get_result()

        if serializer:
            serializer.serialize(res)

        return res
