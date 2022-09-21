from typing import Any, List

from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.base import StatsAnalyzer
from pysolotools.stats.serializers.base import Serializer


class StatsHandler:
    def __init__(self, solo: Solo):
        self.solo = solo

    def handle(
        self,
        analyzers: List[StatsAnalyzer],
        cat_ids: list = None,
        serializer: Serializer = None,
    ) -> dict:
        """
        Handle stats computation and returns dictionary where key is stat class name and value are computed stats.

        Args:
            analyzers (list): list of analyzers.
            cat_ids (list): list of category ids.
            serializer (Serializer): serializer object.

        """

        res = {}
        for i, frame in enumerate(self.solo.frames()):
            for stats_analyzer in analyzers:
                one_frame_stats = stats_analyzer.analyze(frame, cat_ids=cat_ids)

                class_name = stats_analyzer.__class__.__name__
                if class_name not in res:
                    res[class_name] = one_frame_stats
                else:
                    res[class_name] = stats_analyzer.merge(
                        agg_result=res[class_name], frame_result=one_frame_stats
                    )
        if serializer:
            serializer.serialize(res)

        return res
