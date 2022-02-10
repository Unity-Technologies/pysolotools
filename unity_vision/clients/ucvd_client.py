import logging
import os
import tarfile
from datetime import time
from pathlib import Path

import requests
import requests.exceptions
from tqdm import tqdm

from unity_vision.clients.base import DatasetClient
from unity_vision.clients.http_client import HttpClient
from unity_vision.core.auth.basic_auth import BasicAuthenticator
from unity_vision.core.exceptions import (AuthenticationException,
                                          DatasetException,
                                          UnityVisionException)

logger = logging.getLogger(__name__)

BASE_URI_V1 = "https://perception-api.simulation.unity3d.com"
UNITY_AUTH_SA_KEY = "UNITY_AUTH_SA_KEY"
UNITY_AUTH_API_SECRET = "UNITY_AUTH_API_SECRET"
_SDK_VERSION = "v0.0.1"


class UCVDClient(DatasetClient):
    """
    A client for using Unity Computer Vision REST APIs
    """

    def __init__(
        self,
        sa_key=None,
        api_secret=None,
        api_version="v1",
        endpoint=BASE_URI_V1,
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
        self.endpoint = endpoint
        self.api_version = api_version
        self.authenticator = BasicAuthenticator(
            sa_key=self.sa_key, api_secret=self.api_secret
        )

        self.client = HttpClient(
            api_version=self.api_version, authenticator=self.authenticator
        )

        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "",
            "X-User-Agent": f"unity-vision-sdk {_SDK_VERSION})",
        }

    def create_dataset(self, cfg):
        """
        Spec for create dataset

        """
        pass

    def describe_dataset(self, id):
        pass

    def list_datasets(self):
        pass

    def download_dataset(self, run_id) -> str:
        """Download a dataset from remote

        API Spec:
        {{base_url}}/v1/datasets/<run-id>/download
            >> latest archive signed_url
        {{base_url}}/v1/datasets/<run-id>/describe
            >> Describe on a dataset --> Get adapter ids
        {{base_url}}/v1/datasets/<run-id>/adapter/<adapter-id>/download
            >> Downloads a specific run output from an adapter

        Args:
            run_id (str): This is the run_id that identifies a dataset.

        Returns:
            file_path (str): Returns the location where the tarfile was downloaded.
        """

        __entity_uri = Path(self.endpoint, "datasets", run_id)

        req = self._build_request("GET", str(__entity_uri))
        res = self.client.make_request(req)
        dataset_signed_uri = res.content
        return self._download_from_signed_url(dataset_signed_uri, run_id)

    def _download_from_signed_url(self, signed_uri, run_id, file_path=None):
        """Download file from signed URL. Chunked downloads enabled.
        Args:
            signed_uri (str): Signed URI to download dataset (tarfile).
        """
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if file_path is None:
            file_path = f"dataset/{timestr}/{run_id}.tar.gz"
        logger.info(f"Downloading content to {file_path}")
        pbar = tqdm(total=100)
        try:
            with requests.get(signed_uri, stream=True) as r:
                r.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        f.write(chunk)
                        pbar.update(10)
                pbar.close()
        except Exception as e:
            raise UnityVisionException(f"Failed to download file: {str(e)}")

        try:
            self._validate_dataset(file_path)
        except Exception as e:
            raise DatasetException(f"Invalid dataset: {str(e)}")
        return file_path

    def _build_request(self, method, entity_uri):
        """Builds request object

        Args:
            method (str): oneof(GET, PUT, POST, DELETE, PATCH)
            entity_uri (str): API Path for given entity

        Returns:
            object: Returns the resulting request for use further.
        """
        return {"method": method, "url": entity_uri, "headers": self._headers}

    def _validate_dataset(self, file_path: str) -> bool:
        """Validates if a file path is a tarfile. The UCVD datasets are expected to be valid tarfiles.

        TODO: Add checksum support

        Arguments:
            file_path: Path to dataset tarfile

        Returns:
            bool: True if valid tarfile, False otherwise
        """
        return tarfile.is_tarfile(file_path)
