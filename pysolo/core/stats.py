from typing import List

import numpy as np

from pysolo.core.iterators import FramesIterator
from pysolo.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotationDefinition,
    DatasetAnnotations,
    DatasetMetadata,
)


class SoloStats:
    def __init__(
        self,
        data_path: str,
        metadata: DatasetMetadata,
        dataset_annotation: DatasetAnnotations,
        start: int,
        end: int,
    ):
        self.data_path = data_path
        self.metadata = metadata
        self.dataset_annotation = dataset_annotation
        self.start = start
        self.end = end

    def _frame_iter(self):
        return FramesIterator(
            self.data_path,
            self.metadata,
            self.start,
            self.end,
        )

    def get_categories(self):
        categories = {}
        for res in filter(
            lambda ann_def: isinstance(ann_def, BoundingBox2DAnnotationDefinition),
            self.dataset_annotation.annotationDefinitions,
        ):
            for s in res.spec:
                categories[s.label_id] = s.label_name
        return categories

    def get_frame_ids(self):
        ids = []

        for frame in self._frame_iter():
            ids.append(frame.frame)
        return ids

    def get_num_bbox(self, cat_ids: List = None):
        num_bbox = 0

        for frame in self._frame_iter():
            for capture in frame.captures:
                for ann in filter(
                    lambda k: isinstance(k, BoundingBox2DAnnotation),
                    capture.annotations,
                ):
                    annotations = list(
                        filter(
                            lambda x: cat_ids is None or x.labelId in cat_ids,
                            ann.values,
                        )
                    )
                    num_bbox += len(annotations)

        return num_bbox

    def get_bbox_per_img_dist(self, cat_ids: List = None):
        bbox_dist = {}
        for frame in self._frame_iter():
            for capture in frame.captures:
                for ann in filter(
                    lambda k: isinstance(k, BoundingBox2DAnnotation),
                    capture.annotations,
                ):
                    annotations = list(
                        filter(
                            lambda x: cat_ids is None or x.labelId in cat_ids,
                            ann.values,
                        )
                    )
                    num_bbox = len(annotations)
                    if num_bbox in bbox_dist.keys():
                        bbox_dist[num_bbox] += 1
                    else:
                        bbox_dist[num_bbox] = 1

        return bbox_dist

    def get_bbox_heatmap(self, cat_ids: List = None):

        bbox_heatmap = None

        for frame in self._frame_iter():
            for capture in frame.captures:
                if bbox_heatmap is None:
                    w, h = int(capture.dimension[0]), int(capture.dimension[1])
                    bbox_heatmap = np.zeros([h, w, 1])

                for ann in filter(
                    lambda k: isinstance(k, BoundingBox2DAnnotation),
                    capture.annotations,
                ):
                    for v in filter(
                        lambda x: cat_ids is None or x.labelId in cat_ids, ann.values
                    ):
                        bbox = [
                            int(v.origin[0]),
                            int(v.origin[1]),
                            int(v.dimension[0]),
                            int(v.dimension[1]),
                        ]
                        bbox_heatmap[
                            bbox[1] : bbox[1] + bbox[3],
                            bbox[0] : bbox[0] + bbox[2],
                            :,
                        ] += 1

        return bbox_heatmap

    def get_bbox_size_dist(self, cat_ids: List = None):
        bbox_relative_size = []

        for frame in self._frame_iter():
            for capture in frame.captures:
                w, h = capture.dimension[0], capture.dimension[1]
                img_area = w * h
                for ann in filter(
                    lambda k: isinstance(k, BoundingBox2DAnnotation),
                    capture.annotations,
                ):
                    for v in filter(
                        lambda x: cat_ids is None or x.labelId in cat_ids, ann.values
                    ):
                        bbox_area = v.dimension[0] * v.dimension[1]
                        bbox_relative_size.append(np.sqrt(bbox_area / img_area))

        return bbox_relative_size
