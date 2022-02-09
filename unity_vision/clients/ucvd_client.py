import json
import logging
import os
from datetime import time
from pathlib import Path
from tqdm import tqdm
import requests
import requests.exceptions
import contextvars

from unity_vision.clients.http_client import HttpClient
from unity_vision.core.auth.basic_auth import BasicAuthenticator
from unity_vision.core.exceptions import (AuthenticationException,
                                          TimeoutException,
                                          UnityVisionException)
from unity_vision.core.model import UCVDDataset

logger = logging.getLogger(__name__)

UNITY_AUTH_SA_KEY = "UNITY_AUTH_SA_KEY"
UNITY_AUTH_API_SECRET = "UNITY_AUTH_API_SECRET"
_SDK_VERSION = "v0.0.1"
BASE_URI_V1 = "https://perception-api.simulation.unity3d.com"

class UCVDClient:
    """
    A UCVD client necessary to interact with UCVD APIs. Provides functions to download UCVD Datasets.
    """
    def __init__(self,
                 sa_key=None,
                 api_secret=None,
                 api_version="v1",
                 endpoint=BASE_URI_V1, **kwargs):
        """
        Creates and initializes a UCVDClient

        Usage:

        >> client = UCVDClient(
            sa_key="UNITY_AUTH_SA_KEY",
            api_secret="UNITY_AUTH_API_SECRET"
            )
        >> client.health()

        Args:
            api_key (str): API Key. If None, it defaults to the `UNITY_API_KEY` environment variable.
            endpoint (str): Base URI for Unity Computer Vision Dataset APIs.

        """
        if sa_key is None or api_secret is None:
            if UNITY_AUTH_SA_KEY not in os.environ or UNITY_AUTH_API_SECRET not in os.environ:
                raise AuthenticationException(
                    "UNITY_AUTH_SA_KEY and UNITY_AUTH_API_SECRET both must be present."
                )
            self.sa_key = os.environ[UNITY_AUTH_SA_KEY]
            self.api_secret = os.environ[UNITY_AUTH_API_SECRET]
        self.sa_key = sa_key
        self.api_secret = api_secret

        self.api_version = api_version
        self.authenticator = BasicAuthenticator(
            sa_key=self.sa_key,
            api_secret=self.api_secret
        )

        self.client = HttpClient(
            api_version=self.api_version,
            authenticator=self.authenticator)

        self.endpoint = endpoint

        self._headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            "User-Agent": "",
            'X-User-Agent': f'unity-vision-sdk {_SDK_VERSION})',
        }

    def _download_from_signed_url(self, signed_uri, run_id, file_path=None):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if file_path is None:
            file_path = f"{run_id}/dataset/{timestr}/dataset.tar.gz"
        logger.info(f"Downloading content to {file_path}")
        pbar = tqdm(total=100)
        with requests.get(signed_uri, stream=True) as r:
            r.raise_for_status()
            l = len(r.content())
            print(l)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    pbar.update(10)
            pbar.close()

        return file_path

    def get_dataset(self, run_id):
        """"""
        __entity_uri = Path(self.endpoint, "datasets", run_id)

        req = {
            'method': 'GET',
            'uri': __entity_uri,
            'headers': self._headers
        }
        res = self.client.make_request(req)
        dataset_signed_uri = res.content
        return self._download_from_signed_url(dataset_signed_uri, run_id)
