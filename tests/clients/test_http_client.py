import unittest
from http import HTTPStatus
from unittest import mock
from unittest.mock import MagicMock, patch

from unity_vision.clients.http_client import HttpClient

MOCK_HOST = "https://mock-api"
MOCK_SA_KEY = "mock-sa-key"
MOCK_API_SECRET = "mock-api-secret"
TIMEOUT = 180


class TestHttpClient(unittest.TestCase):
    def _mock_response(
        self, status=200, data=None, content="MOCK_CONTENT", raise_for_status=None
    ):
        mock_response = mock.Mock()
        mock_response.raise_for_status = mock.Mock()
        if raise_for_status:
            mock_response.raise_for_status.side_effect = raise_for_status

        mock_response.status_code = status
        mock_response.content = content

        if data:
            mock_response.json = mock.Mock(return_value=data)

        return mock_response

    def _mock_auth_req(self, req):
        headers = req.get("headers")
        headers["Authentication"] = f"Basic xyz"

    @patch("unity_vision.core.auth.basic_auth.BasicAuthenticator")
    @patch("requests.request")
    def test_get(self, mock_get_request, mock_authenticator):
        mock_content = "test-get-response"
        mock_resp = self._mock_response(content=mock_content)
        mock_authenticator.authenticate = MagicMock(return_value=None)
        mock_authenticator.side_effect = self._mock_auth_req
        mock_authenticator.validate = MagicMock(return_value=True)
        mock_get_request.return_value = mock_resp

        client = HttpClient(api_version="v1", authenticator=mock_authenticator)
        req = {"method": "GET", "uri": MOCK_HOST, "headers": {}, "timeout": TIMEOUT}
        res = client.make_request(req)
        assert res.content == mock_content
        mock_get_request.called_only_once()
        mock_get_request.assert_called_with(**req)
        assert res.status_code == HTTPStatus.OK


if __name__ == "__main__":
    unittest.main()
