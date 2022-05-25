from typing import List

import numpy as np

from pysolo.core import (BoundingBox2DAnnotation,
                         BoundingBox2DAnnotationDefinition)
from pysolo.core.iterators import FramesIterator


class SoloStats:

    def __init__(self, iterator: FramesIterator):
        self.frames = iterator

    def _reset_frame_idx(self):
        self.frames.frame_idx = 0

    def get_categories(self):
        categories = {}
        for res in filter(lambda ann_def: isinstance(ann_def, BoundingBox2DAnnotationDefinition), self.solo.annotation_definitions):
            for s in res.spec:
                categories[s.label_id] = s.label_name
        return categories

    def get_frame_ids(self):
        ids = []

        while True:
            try:
                frame = next(self.frames)
                ids.append(frame.frame)
            except StopIteration:
                break

        self._reset_frame_idx()
        return ids

    def get_num_bbox(self, cat_ids: List = None):
        num_bbox = 0
        while True:
            try:
                frame = next(self.frames)
                for f in frame.captures:
                    for ann in filter(
                            lambda k: isinstance(k, BoundingBox2DAnnotation),
                            f.annotations):
                        annotations = list(filter(
                                lambda x: cat_ids is None or x.labelId in cat_ids,
                                ann.values))
                        num_bbox += len(annotations)
            except StopIteration:
                break
        self._reset_frame_idx()

        return num_bbox

    def get_bbox_per_img_dist(self, cat_ids: List = None):
        bbox_dist = {}
        while True:
            try:
                frame = next(self.frames)
                for f in frame.captures:
                    for ann in filter(
                            lambda k: isinstance(k, BoundingBox2DAnnotation),
                            f.annotations):
                        annotations = list(filter(
                                lambda x: cat_ids is None or x.labelId in cat_ids,
                                ann.values))
                        num_bbox = len(annotations)
                        if num_bbox in bbox_dist.keys():
                            bbox_dist[num_bbox] += 1
                        else:
                            bbox_dist[num_bbox] = 1
            except StopIteration:
                break

        self._reset_frame_idx()
        return bbox_dist

    def get_bbox_heatmap(self, cat_ids: List = None):

        bbox_heatmap = None

        while True:
            try:
                frame = next(self.frames)

                for f in frame.captures:
                    if bbox_heatmap is None:
                        w, h = int(f.dimension[0]), int(f.dimension[1])
                        bbox_heatmap = np.zeros([h, w, 1])
                    for ann in filter(
                            lambda k: isinstance(k, BoundingBox2DAnnotation),
                            f.annotations):
                        for v in filter(
                                lambda x: cat_ids is None or x.labelId in cat_ids,
                                ann.values):
                            bbox = [int(v.origin[0]), int(v.origin[1]),
                                    int(v.dimension[0]), int(v.dimension[1])]
                            bbox_heatmap[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2], :] += 1

            except StopIteration:
                break
        self._reset_frame_idx()
        return bbox_heatmap

    def get_bbox_size_dist(self, cat_ids: List = None):
        bbox_relative_size = []
        while True:
            try:
                frame = next(self.frames)
                for f in frame.captures:
                    w, h = f.dimension[0], f.dimension[1]
                    img_area = w * h
                    for ann in filter(
                            lambda k: isinstance(k, BoundingBox2DAnnotation),
                            f.annotations):
                        for v in filter(
                                lambda x: cat_ids is None or x.labelId in cat_ids,
                                ann.values):
                            bbox_area = v.dimension[0] * v.dimension[1]
                            bbox_relative_size.append(np.sqrt(bbox_area / img_area))

            except StopIteration:
                break
        self._reset_frame_idx()
        return bbox_relative_size
