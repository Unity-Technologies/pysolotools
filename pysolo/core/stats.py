import math
from typing import Any, List, Tuple

import numpy as np

from pysolo.core.exceptions import MissingKeypointAnnotatorException
from pysolo.core.iterators import FramesIterator
from pysolo.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotationDefinition,
    DatasetAnnotations,
    DatasetMetadata,
    KeypointAnnotation,
    KeypointAnnotationDefinition,
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

    def _check_keypoint_annotator(self):
        kp = list(
            filter(
                lambda k: isinstance(k, KeypointAnnotationDefinition),
                self.dataset_annotation.annotationDefinitions,
            )
        )
        if not kp:
            raise MissingKeypointAnnotatorException("Keypoint annotations missing")

    def _get_kp_label_dict(self):
        kps = list(
            filter(
                lambda k: isinstance(k, KeypointAnnotationDefinition),
                self.dataset_annotation.annotationDefinitions,
            )
        )[0].template.keypoints
        return {kp.index: kp.label for kp in kps}

    def get_avg_keypoint_per_cat(self):
        self._check_keypoint_annotator()
        kps_label_dict = self._get_kp_label_dict()
        avg_kp_dict = {kp: 0 for kp in kps_label_dict.keys()}
        total_kp_bbox = 0

        for frame in self._frame_iter():
            for capture in frame.captures:
                for ann in filter(
                    lambda k: isinstance(k, KeypointAnnotation),
                    capture.annotations,
                ):
                    for kp_ann in ann.values:
                        total_kp_bbox += 1
                        for kp in kp_ann.keypoints:
                            avg_kp_dict[kp.index] += 1

        return {
            kps_label_dict[key]: avg_kp_dict[key] / total_kp_bbox
            for key in avg_kp_dict.keys()
        }

    @staticmethod
    def _is_torso_visible_or_labeled(kp: List) -> bool:
        torso = []
        for keypoint in filter(
            lambda k: k.index in [2, 5, 8, 11],
            kp,
        ):
            torso.append(keypoint.state)

        if 0 in torso:
            return False
        return True

    @staticmethod
    def _calc_dist(p1: Tuple[Any, Any], p2: Tuple[Any, Any]) -> float:
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    @staticmethod
    def _calc_mid(p1: Tuple[Any, Any], p2: Tuple[Any, Any]):
        return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    @staticmethod
    def _translate_and_scale_xy(x_arr: np.ndarray, y_arr: np.ndarray):

        left_hip, right_hip = (x_arr[11], y_arr[11]), (x_arr[8], y_arr[8])
        left_shoulder, right_shoulder = (x_arr[5], y_arr[5]), (x_arr[2], y_arr[2])

        # Translate all points according to mid_hip being at 0,0
        mid_hip = SoloStats._calc_mid(right_hip, left_hip)
        x_arr = np.where(x_arr > 0.0, x_arr - mid_hip[0], 0.0)
        y_arr = np.where(y_arr > 0.0, y_arr - mid_hip[1], 0.0)

        # Calculate scale factor
        scale = (
            SoloStats._calc_dist(left_shoulder, left_hip)
            + SoloStats._calc_dist(right_shoulder, right_hip)
        ) / 2

        return x_arr / scale, y_arr / scale

    def get_kp_pose_dict(self):
        kps_label_dict = self._get_kp_label_dict()
        kp_pose_dict = {kp: {"x": [], "y": []} for kp in kps_label_dict.keys()}
        for frame in self._frame_iter():
            for capture in frame.captures:
                for ann in filter(
                    lambda k: isinstance(k, KeypointAnnotation),
                    capture.annotations,
                ):
                    for kp_ann in ann.values:
                        if self._is_torso_visible_or_labeled(kp_ann.keypoints):
                            x_loc, y_loc = [], []
                            for kp in kp_ann.keypoints:
                                x_loc.append(kp.location[0])
                                y_loc.append(kp.location[1])
                            x_loc, y_loc = self._translate_and_scale_xy(
                                np.array(x_loc), np.array(y_loc)
                            )

                            idx = 0
                            for xi, yi in zip(x_loc, y_loc):
                                if xi == 0 and yi == 0:
                                    pass
                                elif xi > 2.5 or xi < -2.5 or yi > 2.5 or yi < -2.5:
                                    pass
                                else:
                                    kp_pose_dict[idx]["x"].append(xi)
                                    kp_pose_dict[idx]["y"].append(yi)
                                idx += 1

        return {kps_label_dict[key]: kp_pose_dict[key] for key in kp_pose_dict.keys()}
