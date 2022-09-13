from typing import Any

import numpy as np

from pysolotools.core.models import BoundingBox2DAnnotation, Frame
from pysolotools.stats.analyzers.base import StatsAnalyzer


class BBoxSizeAnalyzer(StatsAnalyzer):
    def analyze(
        self, frame: Frame = None, cat_ids: list = None, **kwargs: Any
    ) -> object:
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
                if cat_ids and v.labelId not in cat_ids:
                    continue
                box_area = v.dimension[0] * v.dimension[1]
                relative_size = np.sqrt(box_area / img_area)
                res.append(relative_size)
        return res

    def merge(self, results, result) -> object:
        results.extend(result)
        return results


class BBoxHeatMapAnalyzer(StatsAnalyzer):
    def analyze(
        self, frame: Frame = None, cat_ids: list = None, **kwargs: Any
    ) -> object:

        """
        Args:
            frame (Frame): metadata of one frame
        Returns:
            bbox_heatmap (np.ndarray): numpy array of size of
            the max sized image in the dataset with values describing
            bbox intensity over the entire dataset images
            at a particular pixel.
        """
        img_dim, bounding_boxes = _frame_bbox_dim(frame)
        bbox_heatmap = np.zeros([img_dim[1], img_dim[0], 1])
        for box in bounding_boxes:
            for v in box.values:
                if cat_ids and v.labelId not in cat_ids:
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

    def merge(self, results, result) -> object:
        return results + result


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
