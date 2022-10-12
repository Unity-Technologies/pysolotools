from typing import Any, Dict, List, Tuple

import numpy as np

from pysolotools.consumers import Solo
from pysolotools.core.models import BoundingBox2DAnnotation, Frame
from pysolotools.stats.analyzers.base import StatsAnalyzer


class BBoxCountStats:
    def __init__(self, solo: Solo):
        self._solo = solo
        self._categories = solo.categories()
        self._labels_to_id = {value: key for key, value in self._categories.items()}
        self._frame_counts = {}
        self._counts = {}

    def get_ids(self):
        return self._categories.keys()

    def get_labels(self):
        return self._labels_to_id.keys()

    def add_counts(self, data: Tuple[int, Dict[int, int]]):
        self._frame_counts[data[0]] = data[1]
        self._counts = {
            x: self._counts.get(x, 0) + data[1].get(x, 0)
            for x in set(self._counts).union(data[1])
        }

    def get_count(self, ids: List[int], frame: int = None) -> int:
        res = 0
        for i in ids:
            res += self._counts.get(i, 0)
        return res

    def get_count_by_label(self, labels: List[str]) -> int:
        ids = []
        for i in labels:
            ids.append(self._labels_to_id.get(i, -1))
        return self.get_count(ids)

    def get_total_count(self):
        return sum(self._counts.values())

    def get_count_per_frame(
        self, frames: List[int], ids: List[int] = None
    ) -> List[int]:
        count_per_frame = {}
        for f in frames:
            total = 0
            if f in self._frame_counts:
                for key in self._frame_counts[f]:
                    if ids and key not in ids:
                        continue
                    total += self._frame_counts[f][key]
            count_per_frame[f] = total

        return count_per_frame

    def get_count_per_frame_by_label(
        self, frames: List[int], labels: List[str]
    ) -> List[int]:
        if not labels:
            return self.get_count_per_frame(frames)

        ids = []
        for i in labels:
            ids.append(self._labels_to_id.get(i, -1))

        return self.get_count_per_frame(frames, ids)


class BBoxCountStatsAnalyzer(StatsAnalyzer):
    def __init__(self, solo: Solo, cat_ids: List = None):
        self._solo = solo
        self._cat_ids = cat_ids
        self._totals = BBoxCountStats(solo)

    def analyze(self, frame: Frame = None, **kwargs: Any) -> Any:
        frame_counts = {}

        for capture in frame.captures:
            for annotation in capture.annotations:
                if isinstance(annotation, BoundingBox2DAnnotation):
                    for v in annotation.values:
                        if self._cat_ids and v.labelId not in self._cat_ids:
                            continue

                        frame_counts[v.labelId] = frame_counts.get(v.labelId, 0) + 1

        return frame.frame, frame_counts

    def merge(self, frame_result: Any, **kwargs: Any):
        self._totals.add_counts(frame_result)

    def get_result(self) -> Any:
        return self._totals


class BBoxSizeStatsAnalyzer(StatsAnalyzer):
    def __init__(self, cat_ids: List = None):
        self._cat_ids = cat_ids
        self._res = []

    def analyze(self, frame: Frame = None, **kwargs: Any) -> List:
        """
        Args:
            frame (Frame): metadata of one frame
        Returns:
            bbox_relative_size_list (list): List of all bbox
             sizes relative to its image size
        """

        img_dim, bounding_boxes = _frame_bbox_dim(frame)
        img_area = img_dim[0] * img_dim[1]
        res = []
        for box in bounding_boxes:
            for v in box.values:
                if self._cat_ids and v.labelId not in self._cat_ids:
                    continue
                box_area = v.dimension[0] * v.dimension[1]
                relative_size = np.sqrt(box_area / img_area)
                res.append(relative_size)
        return res

    def merge(self, frame_result: List, **kwargs):
        """
        Merge computed stats values.
        Args:
            frame_result (list):  result of one frame.

        Returns:
            aggregated stats values.

        """
        self._res.extend(frame_result)

    def get_result(self):
        return self._res


class BBoxHeatMapStatsAnalyzer(StatsAnalyzer):
    def __init__(self, cat_ids: List = None):
        self._cat_ids = cat_ids
        self._res = None

    def analyze(self, frame: Frame = None, **kwargs: Any) -> np.ndarray:

        """
        Args:
            frame (Frame): metadata of one frame
        Returns:
            bbox_heatmap (np.ndarray): numpy array of size of
            the image in the dataset with values describing
            bbox intensity over one frame of dataset.
        """
        img_dim, bounding_boxes = _frame_bbox_dim(frame)
        bbox_heatmap = np.zeros([img_dim[1], img_dim[0], 1])
        for box in bounding_boxes:
            for v in box.values:
                if self._cat_ids and v.labelId not in self._cat_ids:
                    continue
                bbox = [
                    int(v.origin[0]),
                    int(v.origin[1]),
                    int(v.dimension[0]),
                    int(v.dimension[1]),
                ]
                bbox_heatmap[
                    bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
                ] += 1
        return bbox_heatmap

    def merge(self, frame_result: np.ndarray, **kwargs):
        """
        Merge computed stats values.
        Args:
            frame_result (np.ndarray):  result of one frame.

        Returns:
            aggregated stats values.

        """
        if isinstance(self._res, np.ndarray):
            self._res += frame_result
        else:
            self._res = frame_result

    def get_result(self):
        return self._res


def _frame_bbox_dim(frame):
    bounding_boxes = []

    img_dim = [0, 0]
    for capture in frame.captures:
        img_dim[0] = int(capture.dimension[0])
        img_dim[1] = int(capture.dimension[1])

        bounding_boxes.extend(
            filter(
                lambda k: isinstance(k, BoundingBox2DAnnotation),
                capture.annotations,
            )
        )
    return img_dim, bounding_boxes
