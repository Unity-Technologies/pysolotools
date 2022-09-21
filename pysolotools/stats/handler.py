from typing import Dict, List

from pysolotools.consumers import Solo
from pysolotools.stats.serializers.base import Serializer


class StatsHandler:
    def __init__(self, solo: Solo):
        self.solo = solo

    def handle(
        self, analyzers: List[str], cat_ids: List = None, serializer: Serializer = None
    ) -> Dict:
        """
        Handle stats computation and returns dictionary where key is stat class name and value are computed stats.

        Args:
            analyzers (list): list of analyzers.
            cat_ids (list): list of category ids.
            serializer (Serializer): serializer object.

        """
        stats_analyzers = []
        for analyzer in analyzers:
            stats_analyzers.append(AnalyzerFactory.create_analyzer(name=analyzer))

        res = {}
        for i, frame in enumerate(self.solo.frames()):
            for stats_analyzer in stats_analyzers:
                one_frame_stats = stats_analyzer.analyze(
                    frame, cat_ids=cat_ids, solo_data_path=self.solo.data_path
                )

                class_name = stats_analyzer.__class__.__name__
                if class_name not in res:
                    res[class_name] = one_frame_stats
                else:
                    res[class_name] = stats_analyzer.merge(
                        results=res[class_name], result=one_frame_stats
                    )
        if serializer:
            serializer.serialize(res)

        return res
