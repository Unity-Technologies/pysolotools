import base64

from .auth import Authenticator


class BasicAuthenticator(Authenticator):
    """
    BasicAuthenticator works with the Unity Service Account based authentication. It adds a API token to the
    request based on the Service Account Key and the API Secret provided. This can be fetched from the Unity Dashboard.


    The Basic Authentication will be sent as a Authentication header as follows:

        Authorization: Basic <encoded sa_key:api_secret>

    Args:
        sa_key: Service Account Key
        api_secret: API Secret

    Raises:
        ValueError: If the SA Key or the API Secret is not available.
    """
    def __init__(self, sa_key: str, api_secret: str, *args, **kwargs):
        self.sa_key = sa_key
        self.api_secret = api_secret
        self.token = BasicAuthenticator.build_token(sa_key, api_secret)
        self.validate()

    def validate(self) -> None:
        """Validates presence of generated token from sa_key and api_secret.

        Raises:
            ValueError: If the token is not present.
        """
        if self.token is None:
            raise ValueError("Auth token not present. Please make sure to provide valid SA Key and API Secret.")

    def authenticate(self, req) -> None:
        """Add basic authentication header to request

        Args:
            req (dict): The request to add basic auth header to.
        """
        if 'headers' not in req:
            req["headers"] = {}
        headers = req.get('headers')
        headers["Authorization"] = f"Basic {self.token}"

    def authentication_type(self) -> str:
        """Returns authenticator type ('basic')."""
        return Authenticator.AUTH_BASIC

    @staticmethod
    def build_token(sa_key, api_secret):
        return base64.b64encode(
            f"{sa_key}:{api_secret}".encode("utf-8")
        ).decode("utf-8")
