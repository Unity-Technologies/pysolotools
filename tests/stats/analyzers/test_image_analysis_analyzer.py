import os
from unittest.mock import call, create_autospec, patch

import numpy as np
import pytest
from PIL.Image import Image

from pysolotools.core import Frame
from pysolotools.stats.analyzers import (
    PowerSpectrumAnalyzerBase,
    WaveletTransformAnalyzerBase,
)


class TestPowerSpectrumAnalyzer:
    @patch.object(PowerSpectrumAnalyzerBase, "_get_psd1d", autospec=True)
    @patch.object(PowerSpectrumAnalyzerBase, "_get_psd2d", autospec=True)
    @patch.object(PowerSpectrumAnalyzerBase, "_load_img", autospec=True)
    def test_analyze(self, mock_load_img, mock_get_psd2d, mock_get_psd1d):
        mock_frame = create_autospec(spec=Frame, spec_set=True)
        mock_frame.get_file_path.return_value = "some-path"
        analyzer = PowerSpectrumAnalyzerBase()
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
        analyzer = PowerSpectrumAnalyzerBase()
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
        analyzer = PowerSpectrumAnalyzerBase()
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
        analyzer = PowerSpectrumAnalyzerBase()
        result = analyzer._get_psd2d(image=input_array)
        np.testing.assert_allclose(result, expected_result)

    @patch.object(np, "array", autospec=True)
    @patch(
        "pysolotools.stats.analyzers.image_analysis_analyzer.Image.open", autospec=True
    )
    def test_load_img(self, mock_image_open, mock_array):
        mock_image = create_autospec(spec=Image)
        mock_image_open.return_value = mock_image

        analyzer = PowerSpectrumAnalyzerBase()
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
        analyzer = WaveletTransformAnalyzerBase()
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
        analyzer = WaveletTransformAnalyzerBase()
        result = analyzer.merge(agg_result=agg_result, frame_result=[[1], [2], [3]])

        assert result == expected_result
