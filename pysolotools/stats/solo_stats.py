from typing import List

from pysolotools.consumers.solo import Solo
from pysolotools.core.iterators import FramesIterator
from pysolotools.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotationDefinition,
)


class SoloStats:
    def __init__(
        self,
        solo: Solo,
    ):
        self.data_path = solo.data_path
        self.metadata = solo.metadata
        self.dataset_annotation = solo.annotation_definitions
        self.start = solo.start
        self.end = solo.end

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
        return list(map(lambda f: f.frame, self._frame_iter()))

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
