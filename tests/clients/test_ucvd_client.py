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
    mock_datasets_list = [
        {
            "id": "f8407a04-bccd-4934-8cc0-63d7bbdd6b0c",
            "name": "Mock Test Dataset 1",
            "createdAt": "2022-04-29T21:04:05.564332Z",
            "updatedAt": "2022-04-29T21:04:05.564332Z"
        },
        {
            "id": "f8407a04-bccd-4934-8cc0-63d7bbdd6b0c",
            "name": "Test Dataset 2",
            "createdAt": "2022-04-29T21:04:35.564332Z",
            "updatedAt": "2022-04-29T21:04:35.564332Z"
        },
        {
            "id": "f8407a04-bccd-4934-8cc0-63d7bbdd6b0c",
            "name": "Test Dataset 3",
            "createdAt": "2022-04-29T21:04:45.564332Z",
            "updatedAt": "2022-04-29T21:04:45.564332Z"
        }

    ]

    client = UCVDClient(
        project_id=MOCK_PROJECT_ID,
        org_id=MOCK_ORG_ID,
        sa_key=MOCK_SA_KEY,
        api_secret=MOCK_API_SECRET,
    )

    @responses.activate
    def test_list_datasets(self):
        responses.add(responses.GET, f"{API_ENDPOINT}/datasets",
                      json={'results': self.mock_datasets_list}, status=200)

        result = self.client.list_datasets()
        assert result == self.mock_datasets_list

    @responses.activate
    def test_create_dataset(self):
        dataset = {
            "id": "b3eb0caf-52dd-46e5-bf61-ae46bb597e0e",
            "name": "Test Dataset 1",
            "description": "Mock datasets",
            "licenseURI": "string",
            "createdAt": "2022-05-04T07:36:52.233871Z",
            "updatedAt": "2022-05-04T07:36:52.233871Z"
        }
        responses.add(responses.POST, f"{API_ENDPOINT}/datasets", json=dataset, status=200)

        result = self.client.create_dataset(dataset_name="test dataset 1")
        assert result == dataset
