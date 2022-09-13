from typing import Any

import numpy as np

from pysolotools.core.models import BoundingBox2DAnnotation, Frame
from pysolotools.stats.analyzers.base import StatsAnalyzer


class BBoxSizeAnalyzer(StatsAnalyzer):
    def analyze(self, frame: Frame = None, **kwargs: Any) -> object:
        """
        Args:
            frame (Frame): metadata of one frame
        Returns:
            bbox_relative_size_list (list): List of all bbox
             sizes relative to its image size
        """

        bounding_boxes = []

        cap_dim = [0, 0]
        for capture in frame.captures:
            cap_dim[0] = int(capture.dimension[0])
            cap_dim[1] = int(capture.dimension[1])

            bounding_boxes.extend(
                filter(
                    lambda k: isinstance(k, BoundingBox2DAnnotation),
                    capture.annotations,
                )
            )
        img_area = cap_dim[0] * cap_dim[1]
        res = []
        for box in bounding_boxes:
            for v in box.values:
                box_area = v.dimension[0] * v.dimension[1]
                relative_size = np.sqrt(box_area / img_area)
                res.append(relative_size)
        return res


class BBoxHeatMapAnalyzer(StatsAnalyzer):
    def analyze(self, frame: Frame = None, **kwargs: Any) -> object:

        """
        Args:
            frame (Frame): metadata of one frame
        Returns:
            bbox_heatmap (np.ndarray): numpy array of size of
            the max sized image in the dataset with values describing
            bbox intensity over the entire dataset images
            at a particular pixel.
        """
        bounding_boxes = []

        cap_dim = [0, 0]
        for capture in frame.captures:
            cap_dim[0] = int(capture.dimension[0])
            cap_dim[1] = int(capture.dimension[1])

            bounding_boxes.extend(
                filter(
                    lambda k: isinstance(k, BoundingBox2DAnnotation),
                    capture.annotations,
                )
            )
        bbox_heatmap = np.zeros([cap_dim[1], cap_dim[0], 1])
        for box in bounding_boxes:
            for v in box.values:
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
