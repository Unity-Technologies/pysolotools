import logging
import os
import tarfile

import requests
import requests.exceptions
from requests.auth import HTTPBasicAuth

from unity_vision.clients.base import DatasetClient
from unity_vision.core.exceptions import AuthenticationException, UCVDException

logger = logging.getLogger(__name__)

BASE_URI_V1 = 'https://services.api.unity.com/computer-vision-datasets/v1'
UNITY_AUTH_SA_KEY = "UNITY_AUTH_SA_KEY"
UNITY_AUTH_API_SECRET = "UNITY_AUTH_API_SECRET"
_SDK_VERSION = "v0.0.1"


class UCVDClient(DatasetClient):
    """
    A client for using Unity Computer Vision REST APIs
    """

    def __init__(
            self,
            org_id,
            project_id,
            sa_key=None,
            api_secret=None,
            api_version="v1",
            base_uri=BASE_URI_V1,
            **kwargs,
    ):
        """
        Creates and initializes a UCVDClient

        Usage:

        >> client = UCVDClient(
            sa_key="UNITY_AUTH_SA_KEY",
            api_secret="UNITY_AUTH_API_SECRET"
            )

        Args:
            sa_key (str): Unity project service account key. Falls back to UNITY_AUTH_SA_KEY
                            environment variable.
            api_secret (str): API Secret for project. Falls back to UNITY_AUTH_API_SECRET
                                environment variable.
            api_version (str): Version for UCVD APIs being used.
            endpoint (str): Base URI for Unity Computer Vision Dataset APIs.

        Raises:
            AuthenticationException: If Service Account Key and API Secret
                                        are not provided or are invalid.


        """
        self.project_id = project_id
        self.org_id = org_id
        if sa_key is None or api_secret is None:
            if (
                    UNITY_AUTH_SA_KEY not in os.environ
                    or UNITY_AUTH_API_SECRET not in os.environ
            ):
                raise AuthenticationException(
                    "UNITY_AUTH_SA_KEY and UNITY_AUTH_API_SECRET both must be present."
                )
            self.sa_key = os.environ[UNITY_AUTH_SA_KEY]
            self.api_secret = os.environ[UNITY_AUTH_API_SECRET]
        self.sa_key = sa_key
        self.api_secret = api_secret
        self.endpoint = f"{base_uri}/organizations/{self.org_id}/projects/{self.project_id}"
        self.api_version = api_version
        self.auth = HTTPBasicAuth(self.sa_key, self.api_secret)
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "",
            "X-User-Agent": f"unity-vision-sdk {_SDK_VERSION})",
        }

    def create_dataset(self, cfg):
        pass

    def describe_dataset(self, dataset_id):
        pass

    def list_datasets(self):
        entity_uri = f"{self.endpoint}/datasets"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload["results"]

    def download_dataset(self, dataset_id: str, dest_dir: str, chunk_size=1024 ** 2,
                         skip_on_error: bool = True):
        """
        Args:
            dataset_id (str): The dataset id
            dest_dir (str): Destination directory
            chunk_size (int): Chunk size
            skip_on_error (bool): Should skip on error flag

        Returns:
            Downloads the file and saves it to the dest_dir
        """
        pass

    @staticmethod
    def __make_request(
            method,
            url,
            headers=None,
            auth=None,
            params=None,
            body=None,
            data=None
    ):
        session = requests.Session()
        params = params or {}

        try:
            res = session.request(
                method=method,
                url=url,
                headers=headers,
                auth=auth,
                params=params,
                json=body,
                data=data
            )
            res.raise_for_status()
            session.close()
            return res.json()
        except requests.exceptions.HTTPError as re:
            session.close()
            logger.error(str(re))
            raise UCVDException(str(re))

    @staticmethod
    def validate_dataset(file_path: str) -> bool:
        """Validates if a file path is a tarfile. The UCVD datasets are expected to be valid tarfiles.

        TODO: Add checksum support

        Arguments:
            file_path: Path to dataset tarfile

        Returns:
            bool: True if valid tarfile, False otherwise
        """
        return tarfile.is_tarfile(file_path)
