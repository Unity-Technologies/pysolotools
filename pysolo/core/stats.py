import os.path
from typing import List, Tuple

import cv2
import numpy as np
import pywt
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from pysolo.core.iterators import FramesIterator
from pysolo.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotationDefinition,
    BoundingBox2DLabel,
    DatasetAnnotations,
    DatasetMetadata,
    RGBCameraCapture,
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
