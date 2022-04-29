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

    @responses.activate
    def test_list_datasets(self):
        responses.add(responses.GET, f"{API_ENDPOINT}/datasets", json={'results': []}, status=200)
        client = UCVDClient(
            project_id=MOCK_PROJECT_ID,
            org_id=MOCK_ORG_ID,
            sa_key=MOCK_SA_KEY,
            api_secret=MOCK_API_SECRET,
        )
        result = client.list_datasets()
        assert result == []
