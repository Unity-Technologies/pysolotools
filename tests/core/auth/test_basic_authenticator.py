import pytest
from unity_vision.core.auth.auth import Authenticator
from unity_vision.core.auth.basic_auth import BasicAuthenticator

MOCK_SA_KEY = "unity-vision-mock-sa-key"
MOCK_API_SECRET = "unity-vision-mock-api-secret"

def test_basic_authenticator():
    authenticator = BasicAuthenticator(MOCK_SA_KEY, MOCK_API_SECRET)
    token = BasicAuthenticator.build_token(MOCK_SA_KEY, MOCK_API_SECRET)
    assert authenticator is not None
    assert authenticator.authentication_type() == Authenticator.AUTH_BASIC
    assert authenticator.token == token

    req = {
        "headers": {}
    }

    authenticator.authenticate(req)
    assert req["headers"]["Authentication"] == f"Basic {token}"

def test_basic_authenticator_fail():
    with pytest.raises(TypeError) as err:
        BasicAuthenticator(None)

