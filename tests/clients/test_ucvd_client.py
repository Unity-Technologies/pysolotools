import os.path
import unittest

import responses

from pysolo.clients.ucvd_client import UCVDClient
from pysolo.core.models import UCVDDataset

MOCK_PROJECT_ID = "mock-proj-id"
MOCK_ORG_ID = "mock-org-id"
MOCK_SA_KEY = "mock-sa-key"
MOCK_API_SECRET = "mock-api-secret"
MOCK_RUN_ID = "123"
BASE_URI = "https://services.api.unity.com/computer-vision-datasets/v1"
API_ENDPOINT = f"{BASE_URI}/organizations/{MOCK_ORG_ID}/projects/{MOCK_PROJECT_ID}"


class TestUCVDClient(unittest.TestCase):

    mock_dataset = {
        "id": "b3eb0caf-52dd-46e5-bf61-ae46bb597e0e",
        "name": "Test Dataset 1",
        "description": "Mock datasets",
        "licenseURI": "string",
        "createdAt": "2022-05-04T07:36:52.233871Z",
        "updatedAt": "2022-05-04T07:36:52.233871Z",
    }

    list_datasets = {"next": "mock-page-2", "results": [mock_dataset]}

    dataset_archives = {
        "results": [
            {
                "id": "id-1",
                "name": "Sample.tar",
                "type": "FULL",
                "downloadURL": "https://mock-signed-url",
                "state": {"status": "READY"},
            },
            {
                "id": "id-2",
                "name": "Sample-2.tar",
                "type": "FULL",
                "downloadURL": "https://mock-signed-url",
                "state": {"status": "READY"},
            },
        ]
    }

    unauthorized_res = {
        "status": 401,
        "title": "Unauthorized",
        "requestId": "698c0d2f-b413-4061-a49b-3def18880395",
        "detail": "Authentication Failed",
    }

    bad_request = {
        "status": 400,
        "title": "Bad Request",
        "requestId": "mock-id",
        "detail": "Bad Request",
        "details": [],
    }

    not_found = {
        "status": 404,
        "title": "Not Found",
        "requestId": "mock-id",
        "detail": "Not Found",
    }

    client = UCVDClient(
        project_id=MOCK_PROJECT_ID,
        org_id=MOCK_ORG_ID,
        sa_key=MOCK_SA_KEY,
        api_secret=MOCK_API_SECRET,
    )

    @responses.activate
    def test_list_datasets(self):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets",
            json={"results": [self.mock_dataset]},
            status=200,
        )

        result = self.client.list_datasets()
        assert result == [self.mock_dataset]

    @responses.activate
    def test_iterate_datasets(self):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets",
            json=self.list_datasets,
            status=200,
        )

        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets?next=mock-page-2",
            json={"results": [self.mock_dataset]},
            status=200,
        )

        count = 0
        for dataset in self.client.iterate_datasets():
            count += 1
            self.assertIsNotNone(dataset)
            self.assertIsInstance(dataset, UCVDDataset)
            self.assertEqual(dataset.name, self.mock_dataset["name"])
        self.assertEqual(count, 2)

    @responses.activate
    def test_list_datasets_unauthorized(self):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets",
            json=self.unauthorized_res,
            status=403,
        )

        with self.assertRaises(Exception) as context:
            self.client.list_datasets()
            self.assertEqual(context.exception.response.status_code, 403)
            self.assertTrue("Forbidden" in str(context.exception))

    @responses.activate
    def test_create_dataset(self):
        responses.add(
            responses.POST,
            f"{API_ENDPOINT}/datasets",
            json=self.mock_dataset,
            status=200,
        )

        result = self.client.create_dataset(dataset_name=self.mock_dataset["name"])
        assert result == self.mock_dataset

    @responses.activate
    def test_create_dataset_bad_request(self):
        responses.add(
            responses.POST,
            f"{API_ENDPOINT}/datasets",
            json=self.mock_dataset,
            status=400,
        )
        with self.assertRaises(Exception) as context:
            self.client.create_dataset(dataset_name=self.mock_dataset["name"])
            self.assertEqual(context.exception.response.status_code, 400)
            self.assertTrue("Forbidden" in str(context.exception))

    @responses.activate
    def test_describe_dataset(self):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}",
            json=self.mock_dataset,
            status=200,
        )

        result = self.client.describe_dataset(dataset_id=self.mock_dataset["id"])
        assert result == self.mock_dataset

    @responses.activate
    def test_describe_dataset_not_found(self):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}",
            json=self.not_found,
            status=404,
        )
        with self.assertRaises(Exception) as context:
            self.client.describe_dataset(dataset_id=self.mock_dataset["id"])
            self.assertEqual(context.exception.response.status_code, 404)
            self.assertTrue("Forbidden" in str(context.exception))

    @responses.activate
    def test_iterate_dataset_archive(self):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}/archives",
            json=self.dataset_archives,
            status=200,
        )

        iterator = self.client.iterate_dataset_archives(
            dataset_id=self.mock_dataset["id"]
        )
        assert next(iterator).id == "id-1"
        assert next(iterator).id == "id-2"

    @unittest.mock.patch(
        "pysolo.clients.ucvd_client.UCVDClient.iterate_dataset_attachments",
        autospec=True,
    )
    @responses.activate
    def test_download_dataset_archives_success(self, mocked_iterate_dataset_archives):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}/archives",
            json=self.dataset_archives,
            status=200,
        )
        with open("tests/data/Sample.tar", "rb") as f:
            responses.add(
                responses.GET,
                "https://mock-signed-url",
                stream=True,
                body=f.read(),
                status=200,
                content_type="application/x-tar",
            )
        self.client.download_dataset_archives(
            dataset_id=self.mock_dataset["id"], dest_dir="test_data"
        )
        for archive in self.client.iterate_dataset_archives(self.mock_dataset["id"]):
            assert os.path.exists(f"test_data/{archive.name}")
            os.remove(f"test_data/{archive.name}")

    @unittest.mock.patch(
        "pysolo.clients.ucvd_client.UCVDClient.iterate_dataset_attachments",
        autospec=True,
    )
    @responses.activate
    def test_download_dataset_archives_failure_unauthorized(
        self, mocked_iterate_dataset_archives
    ):
        responses.add(
            responses.GET,
            f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}/archives",
            json=self.unauthorized_res,
            status=403,
        )

        with self.assertRaises(Exception) as context:
            self.client.download_dataset_archives(
                dataset_id=self.mock_dataset["id"], dest_dir="test_data"
            )

        self.assertEqual(context.exception.response.status_code, 403)
        self.assertTrue("Forbidden" in str(context.exception))
