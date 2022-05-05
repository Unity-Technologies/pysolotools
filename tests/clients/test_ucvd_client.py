import os.path
import unittest

import responses

from unity_vision.clients.ucvd_client import UCVDClient

MOCK_PROJECT_ID = "mock-proj-id"
MOCK_ORG_ID = "mock-org-id"
MOCK_SA_KEY = "mock-sa-key"
MOCK_API_SECRET = "mock-api-secret"
MOCK_RUN_ID = "123"
BASE_URI = 'https://services.api.unity.com/computer-vision-datasets/v1'
API_ENDPOINT = f"{BASE_URI}/organizations/{MOCK_ORG_ID}/projects/{MOCK_PROJECT_ID}"


class TestUCVDClient(unittest.TestCase):

    mock_dataset = {
        "id": "b3eb0caf-52dd-46e5-bf61-ae46bb597e0e",
        "name": "Test Dataset 1",
        "description": "Mock datasets",
        "licenseURI": "string",
        "createdAt": "2022-05-04T07:36:52.233871Z",
        "updatedAt": "2022-05-04T07:36:52.233871Z"
    }

    dataset_archives = {
        "results": [
            {
                "id": "id-1",
                "name": "Sample.tar",
                "type": "FULL",
                "downloadURL": "https://mock-signed-url",
                "state": {
                    "status": "READY"
                }
            },
            {
                "id": "id-2",
                "name": "Sample-2.tar",
                "type": "FULL",
                "downloadURL": "https://mock-signed-url",
                "state": {
                    "status": "READY"
                },
            }
        ]
    }

    client = UCVDClient(
        project_id=MOCK_PROJECT_ID,
        org_id=MOCK_ORG_ID,
        sa_key=MOCK_SA_KEY,
        api_secret=MOCK_API_SECRET,
    )

    @responses.activate
    def test_list_datasets(self):
        responses.add(responses.GET, f"{API_ENDPOINT}/datasets",
                      json={'results': [self.mock_dataset]}, status=200)

        result = self.client.list_datasets()
        assert result == [self.mock_dataset]

    @responses.activate
    def test_create_dataset(self):
        responses.add(responses.POST, f"{API_ENDPOINT}/datasets", json=self.mock_dataset, status=200)

        result = self.client.create_dataset(dataset_name=self.mock_dataset["name"])
        assert result == self.mock_dataset

    @responses.activate
    def test_describe_dataset(self):
        responses.add(responses.GET, f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}",
                      json=self.mock_dataset, status=200)

        result = self.client.describe_dataset(dataset_id=self.mock_dataset["id"])
        assert result == self.mock_dataset

    @responses.activate
    def test_iterate_dataset_archive(self):
        responses.add(responses.GET, f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}/archives",
                      json=self.dataset_archives, status=200)

        iterator = self.client.iterate_dataset_archives(dataset_id=self.mock_dataset['id'])
        assert next(iterator).id == "id-1"
        assert next(iterator).id == "id-2"

    @unittest.mock.patch("unity_vision.clients.ucvd_client.UCVDClient.iterate_dataset_attachments", autospec=True)
    @responses.activate
    def test_download_dataset_archives(self, mocked_iterate_dataset_archives):
        responses.add(responses.GET, f"{API_ENDPOINT}/datasets/{self.mock_dataset['id']}/archives",
                      json=self.dataset_archives, status=200)
        with open('data/Sample.tar', 'rb') as f:
            responses.add(
                responses.GET,
                "https://mock-signed-url",
                stream=True,
                body=f.read(),
                status=200,
                content_type="application/x-tar"
            )
        self.client.download_dataset_archives(dataset_id=self.mock_dataset["id"], dest_dir="test_data")
        for archive in self.client.iterate_dataset_archives(self.mock_dataset["id"]):
            assert os.path.exists(f"test_data/{archive.name}")
            os.remove(f"test_data/{archive.name}")
