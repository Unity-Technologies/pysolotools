import unittest
from unittest.mock import MagicMock, patch

from unity_vision.clients.ucvd_client import UCVDClient

MOCK_SA_KEY = "mock-sa-key"
MOCK_API_SECRET = "mock-api-secret"
MOCK_RUN_ID = "123"


class TestUCVDClient(unittest.TestCase):
    mock_dataset_path = "test/file-path"

    @patch("unity_vision.clients.ucvd_client.UCVDClient._download_from_signed_url")
    @patch("unity_vision.clients.http_client.HttpClient.make_request")
    def test_get_dataset(self, mocked_make_request, mocked__download_from_signed_url):
        mocked_http_client = MagicMock()
        mocked__download_from_signed_url.return_value = self.mock_dataset_path

        with patch("unity_vision.clients.http_client.HttpClient", mocked_http_client):
            client = UCVDClient(
                sa_key=MOCK_SA_KEY,
                api_secret=MOCK_API_SECRET,
            )
            dataset_path = client.download_dataset("run-id")
            mocked_make_request.assert_called_once()
            assert dataset_path == self.mock_dataset_path
