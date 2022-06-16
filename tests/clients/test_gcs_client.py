import unittest
from unittest.mock import patch

from google.cloud import exceptions

from pysolotools.clients.gcs_client import GCSClient
from pysolotools.core.exceptions import DatasetNotFoundException


class TestGCSClient(unittest.TestCase):
    @patch("google.cloud.storage.Client")
    def test_download_directory_success(self, patched_client):
        gcs_client = GCSClient(client=patched_client)
        status = gcs_client.download_directory("gs://bucket/path", "test-path")
        assert status

    @patch("google.cloud.storage.Client")
    def test_download_directory_failure_client_error(self, patched_client):
        patched_client.bucket.side_effect = exceptions.ClientError(
            "Something went wrong"
        )
        gcs_client = GCSClient(client=patched_client)
        status = gcs_client.download_directory("gs://bucket/path", "test-path")
        self.assertFalse(status)

    @patch("google.cloud.storage.Client")
    def test_download_directory_failure_unauthorized(self, patched_client):
        patched_client.bucket.side_effect = exceptions.Unauthorized(
            "Unauthorized access"
        )
        gcs_client = GCSClient(client=patched_client)
        status = gcs_client.download_directory("gs://bucket/path", "test-path")
        self.assertFalse(status)

    @patch("google.cloud.storage.Client")
    def test_download_directory_failure_not_found(self, patched_client):
        patched_client.bucket.side_effect = exceptions.NotFound("404")
        gcs_client = GCSClient(client=patched_client)
        with self.assertRaises(DatasetNotFoundException):
            gcs_client.download_directory("gs://bucket/path", "test-path")
