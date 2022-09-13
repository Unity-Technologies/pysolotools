import math
import os
from typing import Any, List, Tuple

import cv2
import numpy as np
import pywt
from PIL import Image
from pysolotools.consumers.solo import Solo
from pysolotools.core.exceptions import MissingKeypointAnnotatorException
from pysolotools.core.iterators import FramesIterator
from pysolotools.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotationDefinition,
    BoundingBox2DLabel,
    KeypointAnnotation,
    KeypointAnnotationDefinition,
    RGBCameraCapture,
)
from scipy import ndimage
from tqdm import tqdm


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

    def get_wt_coeffs_var(self):

        images = []

        for frame in self._frame_iter():
            images.append(
                os.path.join(self.data_path, frame.get_file_path(RGBCameraCapture))
            )

        horizontal_coeff, vertical_coeff, diagonal_coeff = [], [], []

        for img in tqdm(images):
            im = Image.open(img).convert("L")
            _, (cH, cV, cD) = pywt.dwt2(im, "haar", mode="periodization")
            horizontal_coeff.append(np.array(cH).var())
            vertical_coeff.append(np.array(cV).var())
            diagonal_coeff.append(np.array(cD).var())

        return horizontal_coeff, vertical_coeff, diagonal_coeff

    @staticmethod
    def _get_psd2d(image: np.ndarray) -> np.ndarray:

        h, w = image.shape
        fourier_image = np.fft.fft2(image)
        N = h * w * 2
        psd2d = (1 / N) * np.abs(fourier_image) ** 2
        psd2d = np.fft.fftshift(psd2d)
        return psd2d

    @staticmethod
    def _get_psd1d(psd_2d: np.ndarray) -> np.ndarray:

        h = psd_2d.shape[0]
        w = psd_2d.shape[1]
        wc = w // 2
        hc = h // 2

        # create an array of integer radial distances from the center
        y, x = np.ogrid[-h // 2 : h // 2, -w // 2 : w // 2]
        r = np.hypot(x, y).astype(int)
        idx = np.arange(0, min(wc, hc))
        psd_1d = ndimage.sum(psd_2d, r, index=idx)
        return psd_1d

    @staticmethod
    def _load_img(img_path: str):
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = img.convert("L")
        return np.array(img)

    def get_average_psd_1d(self):

        img_array = []
        for frame in self._frame_iter():
            img = self._load_img(
                os.path.join(self.data_path, frame.get_file_path(RGBCameraCapture))
            )
            img_array.append(img)

        total_psd_1d = []
        max_len = float("-inf")
        for image in tqdm(img_array):
            psd_2d = self._get_psd2d(image)
            psd_1d = self._get_psd1d(psd_2d)
            max_len = max(max_len, len(psd_1d))
            total_psd_1d.append(psd_1d)

        for i in range(len(total_psd_1d)):
            if len(total_psd_1d[i]) < max_len:
                _len = max_len - len(total_psd_1d[i])
                nan_arr = np.empty(_len)
                nan_arr[:] = np.nan
                total_psd_1d[i] = np.append(total_psd_1d[i], nan_arr)

        total_psd_1d = np.asarray(total_psd_1d, dtype=float)

        avg_psd_1d = np.nanmean(total_psd_1d, axis=0)
        std_psd_1d = np.nanstd(total_psd_1d, axis=0)

        return avg_psd_1d, std_psd_1d

    @staticmethod
    def _laplacian_img(img_path: str) -> np.ndarray:
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = laplacian.astype("float")
        return laplacian

    @staticmethod
    def get_bbox_var_laplacian(
        laplacian: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        bbox_var = laplacian[y : y + h, x : x + w]
        return np.nanvar(bbox_var)

    @staticmethod
    def _get_bbox_fg_bg_var_laplacian(
        laplacian: np.ndarray, annotations: List[BoundingBox2DLabel]
    ) -> Tuple[List, np.ndarray]:

        bbox_var_lap = []
        img_laplacian = laplacian

        for ann in annotations:
            x, y, w, h = [
                int(ann.origin[0]),
                int(ann.origin[1]),
                int(ann.dimension[0]),
                int(ann.dimension[1]),
            ]
            bbox_area = w * h
            if bbox_area >= 1200:  # ignoring small bbox sizes
                bbox_var = SoloStats.get_bbox_var_laplacian(
                    img_laplacian, int(x), int(y), int(w), int(h)
                )
                img_laplacian[int(y) : int(y + h), int(x) : int(x + w)] = np.nan
                bbox_var_lap.append(bbox_var)

        img_var_laplacian = np.nanvar(img_laplacian)

        return bbox_var_lap, img_var_laplacian

    def get_var_laplacian(self):
        bg_vars, fg_vars = [], []
        for frame in self._frame_iter():
            laplacian = self._laplacian_img(
                os.path.join(self.data_path, frame.get_file_path(RGBCameraCapture))
            )

            for capture in filter(
                lambda k: isinstance(k, RGBCameraCapture),
                frame.captures,
            ):
                bbox_anns = list(
                    filter(
                        lambda k: isinstance(k, BoundingBox2DAnnotation),
                        capture.annotations,
                    )
                )[0].values
                bbox_var_lap, img_var_laplacian = self._get_bbox_fg_bg_var_laplacian(
                    laplacian, bbox_anns
                )
                bg_vars.append(img_var_laplacian)
                fg_vars.extend(bbox_var_lap)

        return bg_vars, fg_vars
