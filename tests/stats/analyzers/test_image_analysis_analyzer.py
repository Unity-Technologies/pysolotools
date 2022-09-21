import os
from unittest.mock import call, create_autospec, patch

import numpy as np
import pytest
from PIL.Image import Image

from pysolotools.core import (
    BoundingBox2DAnnotation,
    Frame,
    RGBCameraCapture,
    SemanticSegmentationAnnotation,
)
from pysolotools.core.models import BoundingBox2DLabel
from pysolotools.stats.analyzers import (
    LaplacianAnalyzer,
    PowerSpectrumAnalyzer,
    WaveletTransformAnalyzer,
)


class TestPowerSpectrumAnalyzer:
    @patch.object(PowerSpectrumAnalyzer, "_get_psd1d", autospec=True)
    @patch.object(PowerSpectrumAnalyzer, "_get_psd2d", autospec=True)
    @patch.object(PowerSpectrumAnalyzer, "_load_img", autospec=True)
    def test_analyze(self, mock_load_img, mock_get_psd2d, mock_get_psd1d):
        mock_frame = create_autospec(spec=Frame, spec_set=True)
        mock_frame.get_file_path.return_value = "some-path"
        analyzer = PowerSpectrumAnalyzer()
        result = analyzer.analyze(frame=mock_frame, solo_data_path="some-solo-path")

        mock_load_img.assert_called_once_with(
            os.path.join("some-solo-path", "some-path")
        )
        mock_get_psd2d.assert_called_once_with(mock_load_img.return_value)
        mock_get_psd1d.assert_called_once_with(mock_get_psd2d.return_value)
        assert result == [mock_get_psd1d.return_value]

    @pytest.mark.parametrize(
        "agg_result, expected_result",
        [
            (None, ["some-frame"]),
            ([], ["some-frame"]),
            (["some-result"], ["some-result", "some-frame"]),
        ],
    )
    def test_merge(self, agg_result, expected_result):
        analyzer = PowerSpectrumAnalyzer()
        result = analyzer.merge(agg_result=agg_result, frame_result=["some-frame"])

        assert result == expected_result

    @pytest.mark.parametrize(
        "input_array, expected_result",
        [
            (np.array([[1, 2, 3], [4, 5, 6]], np.int32), np.array([6], np.int32)),
            (np.array([[1], [4]], np.int32), np.array([], np.float64)),
        ],
    )
    def test_get_psd1d(self, input_array, expected_result):
        analyzer = PowerSpectrumAnalyzer()
        result = analyzer._get_psd1d(psd_2d=input_array)

        np.testing.assert_array_equal(result, expected_result)

    @pytest.mark.parametrize(
        "input_array, expected_result",
        [
            (
                np.array([[1, 2, 3], [4, 5, 6]], np.int32),
                np.array([[0, 6.75, 0], [1, 36.75, 1]], np.float64),
            ),
        ],
    )
    def test_get_psd2d(self, input_array, expected_result):
        analyzer = PowerSpectrumAnalyzer()
        result = analyzer._get_psd2d(image=input_array)
        np.testing.assert_allclose(result, expected_result)

    @patch.object(np, "array", autospec=True)
    @patch(
        "pysolotools.stats.analyzers.image_analysis_analyzer.Image.open", autospec=True
    )
    def test_load_img(self, mock_image_open, mock_array):
        mock_image = create_autospec(spec=Image)
        mock_image_open.return_value = mock_image

        analyzer = PowerSpectrumAnalyzer()
        result = analyzer._load_img(img_path="some-path")

        mock_image_open.assert_called_once_with("some-path")
        mock_image.convert.assert_has_calls([call("RGB"), call().convert("L")])
        mock_array.assert_called_once_with(
            mock_image.convert.return_value.convert.return_value
        )
        assert result == mock_array.return_value


class TestWaveletTransformAnalyzer:
    @patch(
        "pysolotools.stats.analyzers.image_analysis_analyzer.pywt.dwt2", autospec=True
    )
    @patch(
        "pysolotools.stats.analyzers.image_analysis_analyzer.Image.open", autospec=True
    )
    def test_analyze(self, mock_image_open, mock_dwt2):
        mock_frame = create_autospec(spec=Frame, spec_set=True)
        mock_frame.get_file_path.return_value = "some-path"
        mock_image = create_autospec(spec=Image)
        mock_image_open.return_value = mock_image
        mock_dwt2.return_value = ("foo", (1, 2, 3))
        analyzer = WaveletTransformAnalyzer()
        result = analyzer.analyze(frame=mock_frame, solo_data_path="some-solo-path")

        mock_image_open.assert_called_once_with(
            os.path.join("some-solo-path", "some-path")
        )
        mock_image.convert.assert_called_once_with("L")
        mock_dwt2.assert_called_once_with(
            mock_image.convert.return_value, "haar", mode="periodization"
        )
        assert result == [[1], [2], [3]]

    @pytest.mark.parametrize(
        "agg_result, expected_result",
        [(None, [[1], [2], [3]]), ([[4], [5], [6]], [[4, 1], [5, 2], [6, 3]])],
    )
    def test_merge(self, agg_result, expected_result):
        analyzer = WaveletTransformAnalyzer()
        result = analyzer.merge(agg_result=agg_result, frame_result=[[1], [2], [3]])

        assert result == expected_result


class TestLaplacianAnalyzer:
    @patch.object(LaplacianAnalyzer, "_get_bbox_fg_bg_var_laplacian", autospec=True)
    @patch.object(LaplacianAnalyzer, "_laplacian_img", autospec=True)
    def test_analyzer(self, mock_laplacian_img, mock_bbox_laplacian):
        mock_bbox_laplacian.return_value = ("bbox_var_laps", "img_var_lap")
        mock_frame = create_autospec(spec=Frame(1, 2, 3))
        mock_frame.get_file_path.return_value = "some-path"
        rgb_camera_capture = self._create_rgb_camera_capture()
        mock_frame.captures = [
            rgb_camera_capture,
            self._create_semantic_segmentation_annotation(),
        ]

        analyzer = LaplacianAnalyzer()
        result = analyzer.analyze(frame=mock_frame, solo_data_path="some-solo-path")

        mock_laplacian_img.assert_called_once_with(
            os.path.join("some-solo-path", "some-path")
        )
        mock_bbox_laplacian.assert_called_once_with(
            mock_laplacian_img.return_value, rgb_camera_capture.annotations[0].values
        )
        assert result == ["bbox_var_laps", ["img_var_lap"]]

    @pytest.mark.parametrize(
        "agg_result, expected_result",
        [
            (None, [[1], [2]]),
            ([], [[1], [2]]),
            ([[3], [4]], [[3, 1], [4, 2]]),
        ],
    )
    def test_merge(self, agg_result, expected_result):
        analyzer = LaplacianAnalyzer()
        result = analyzer.merge(agg_result=agg_result, frame_result=[[1], [2]])

        assert result == expected_result

    @patch("pysolotools.stats.analyzers.image_analysis_analyzer.cv2", autospec=True)
    def test_laplacian_img(self, mock_cv2):
        analyzer = LaplacianAnalyzer()
        result = analyzer._laplacian_img(img_path="some-path")

        mock_cv2.imread.assert_called_once_with("some-path")
        mock_cv2.cvtColor.assert_called_once_with(
            mock_cv2.imread.return_value, mock_cv2.COLOR_BGR2GRAY
        )
        mock_cv2.Laplacian.assert_called_once_with(
            mock_cv2.cvtColor.return_value, mock_cv2.CV_64F
        )
        mock_cv2.Laplacian.return_value.astype.assert_called_once_with("float")
        assert result == mock_cv2.Laplacian.return_value.astype.return_value

    @pytest.mark.parametrize(
        "input_laplacian, input_coordinates, expected_result",
        [
            (
                np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1]], np.int32
                ),
                {"x": 1, "y": 2, "w": 1, "h": 2},
                0.25,
            ),
        ],
    )
    def test_get_bbox_var_laplacian(
        self, input_laplacian, input_coordinates, expected_result
    ):
        analyzer = LaplacianAnalyzer()
        result = analyzer._get_bbox_var_laplacian(
            laplacian=input_laplacian, **input_coordinates
        )
        assert result == expected_result

    @patch.object(LaplacianAnalyzer, "_get_bbox_var_laplacian", autospec=True)
    def test_get_bbox_fg_bg_var_laplacian(self, mock_bbox_laplacian):
        mock_bbox_laplacian.return_value = 0.25
        input_laplacian = np.ones((1200, 4))
        input_annotations = [
            BoundingBox2DLabel(
                labelName="some-label",
                origin=[9, 10],
                dimension=[100, 50],
                instanceId=1,
                labelId=2,
            )
        ]
        analyzer = LaplacianAnalyzer()
        result = analyzer._get_bbox_fg_bg_var_laplacian(
            laplacian=input_laplacian, annotations=input_annotations
        )
        mock_bbox_laplacian.assert_called_once_with(input_laplacian, 9, 10, 100, 50)
        assert result == ([0.25], 0.0)

    @staticmethod
    def _create_semantic_segmentation_annotation() -> SemanticSegmentationAnnotation:
        return SemanticSegmentationAnnotation(
            None, None, None, None, None, None, None, None, None
        )

    @staticmethod
    def _create_rgb_camera_capture() -> RGBCameraCapture:
        return RGBCameraCapture(
            rotation=[1],
            velocity=[2],
            acceleration=[3],
            filename="foo",
            dimension=[4],
            projection="",
            matrix=[5],
            annotations=[
                BoundingBox2DAnnotation(
                    "foo",
                    "bar",
                    "1",
                    "",
                    values=[
                        BoundingBox2DLabel(
                            labelName="some-label",
                            origin=[9],
                            dimension=[10],
                            instanceId=1,
                            labelId=2,
                        )
                    ],
                    extra_data=None,
                )
            ],
            description="",
            id="",
            imageFormat="",
            position=[],
            type="",
        )
