import json
import os
from typing import List

import numpy as np

TOTAL_TOKEN = "_TOTAL"
STATS_PATH_NAME = "stats"
BBOX_STATS_NAME = "stats.bounding_boxes.json"
BBOX_HEATMAP_NAME = "stats.bounding_boxes.heatmap.npz"


class SoloStats:
    def __init__(self, solo_path):
        path = os.path.join(solo_path, STATS_PATH_NAME)
        with open(os.path.join(path, BBOX_STATS_NAME), "r") as f:
            bbox_stats = json.load(f)

        self._stats = bbox_stats["stats"][0]
        self._id_to_labels = self._stats["id_to_label"]
        self._label_to_id = self._stats["label_to_id"]
        self._id_to_count = self._stats["id_to_count"]
        self._frame_counts = bbox_stats["frame_counts"]
        self._relative_sizes = self._stats["relative_sizes"]
        self._heatmap = np.load(os.path.join(path, BBOX_HEATMAP_NAME))["arr_0"]

    @property
    def total_object_count(self) -> int:
        return self._id_to_count[TOTAL_TOKEN]

    @property
    def category_labels(self) -> List[str]:
        return [*self._label_to_id.keys()]

    @property
    def category_ids(self) -> List[int]:
        return [int(x) for x in self._id_to_labels.keys()]

    @property
    def frame_ids(self) -> List[int]:
        return [int(x) for x in self._frame_counts.keys()]

    def get_object_count(self, labels: List[str] = None) -> int:
        if labels is None or labels == [] or TOTAL_TOKEN in labels:
            return self._id_to_count[TOTAL_TOKEN]
        else:
            return sum([x[1] for x in self._id_to_count.items() if x[0] in labels])

    def get_object_count_by_id(self, ids: List[int] = None) -> int:
        cat_labels = self._to_label_list(ids)
        return self.get_object_count(cat_labels)

    def _to_label_list(self, category_ids: List[int] = None):
        if category_ids is None:
            return None
        else:
            return [
                item[1]
                for item in self._id_to_labels.items()
                if int(item[0]) in category_ids
            ]

    def get_bbox_per_img_dist_by_ids(self, category_ids: List[int] = None):
        cat_labels = self._to_label_list(category_ids)
        return self.get_bbox_per_img_dist_by_labels(cat_labels)

    def get_bbox_per_img_dist_by_labels(self, category_ids: List[str] = None) -> dict:
        bboxes = {}
        for frame in self._frame_counts:
            if category_ids is None or TOTAL_TOKEN in category_ids:
                bboxes[frame] = self._frame_counts[frame][TOTAL_TOKEN]
            else:
                bboxes[frame] = sum(
                    [
                        x[1]
                        for x in self._frame_counts[frame].items()
                        if x[0] in category_ids
                    ]
                )

        return bboxes

    def get_bbox_heatmap_by_labels(self, category_ids: List[str] = None) -> np.ndarray:
        if category_ids is None or category_ids == [] or TOTAL_TOKEN in category_ids:
            return self._heatmap[:, :, len(self._id_to_labels)]
        else:
            hmap = None
            as_list = [*self._label_to_id.keys()]
            for i in category_ids:
                idx = as_list.index(i)

                if hmap is None:
                    hmap = self._heatmap[:, :, idx]
                else:
                    hmap += self._heatmap[:, :, idx]

            return hmap

    def get_bbox_size_dist_by_labels(self, category_labels: List[str] = None):
        if (
            category_labels is None
            or category_labels == []
            or TOTAL_TOKEN in category_labels
        ):
            return self._relative_sizes[TOTAL_TOKEN]
        else:
            return [
                x[1] for x in self._relative_sizes.items() if x[0] in category_labels
            ]
