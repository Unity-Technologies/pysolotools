from typing import List

import numpy as np

from pysolo.core import BoundingBox2DAnnotation
from pysolo.core.iterators import FramesIterator


class SoloStats:

    def __init__(self, iterator: FramesIterator):
        self.frames = iterator

    def _reset_frame_idx(self):
        self.frames.frame_idx = 0

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
                    for k in f.annotations:
                        if isinstance(k, BoundingBox2DAnnotation):
                            for v in k.values:
                                if cat_ids:
                                    if v.labelId in cat_ids:
                                        num_bbox += 1
                                else:
                                    num_bbox += 1
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
                    for k in f.annotations:
                        if isinstance(k, BoundingBox2DAnnotation):
                            num_bbox = 0
                            for v in k.values:
                                if cat_ids:
                                    if v.labelId in cat_ids:
                                        num_bbox += 1
                                else:
                                    num_bbox += 1
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
                    for k in f.annotations:
                        if isinstance(k, BoundingBox2DAnnotation):
                            for v in k.values:
                                bbox = [int(v.origin[0]), int(v.origin[1]),
                                        int(v.dimension[0]), int(v.dimension[1])]
                                if cat_ids:
                                    if v.labelId in cat_ids:
                                        bbox_heatmap[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2], :] += 1
                                else:
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
                    for k in f.annotations:
                        if isinstance(k, BoundingBox2DAnnotation):
                            for v in k.values:
                                bbox_area = v.dimension[0] * v.dimension[1]
                                if cat_ids:
                                    if v.labelId in cat_ids:
                                        bbox_relative_size.append(np.sqrt(bbox_area / img_area))
                                else:
                                    bbox_relative_size.append(np.sqrt(bbox_area / img_area))

            except StopIteration:
                break
        self._reset_frame_idx()
        return bbox_relative_size
