import json
import logging
import os
from pathlib import Path

import requests
import requests.exceptions

from unity_vision.core.exceptions import (AuthenticationException,
                                          TimeoutException,
                                          UnityVisionException)
from unity_vision.core.model import Dataset, UCVDDataset

logger = logging.getLogger(__name__)

UNITY_API_KEY = "UNITY_API_KEY"
_SDK_VERSION="v0.0.1"
BASE_URI_V1 = "https://perception-api.simulation.unity3d.com"


class UCVDClient:
    """
    A UCVD client necessary to interact with UCVD APIs. Provides functions to download UCVD Datasets.
    """
    def __init__(self, api_key, endpoint=BASE_URI_V1, **kwargs):
        """
        Creates and initializes a UCVDClient

        Usage:

        >>> client = UCVDClient("<UNITY_API_KEY>")

        Args:
            api_key (str): API Key. If None, it defaults to the `UNITY_API_KEY` environment variable.
            endpoint (str): Base URI for Unity Computer Vision Dataset APIs.

        """
        if api_key is None:
            if "UNITY_API_KEY" not in os.environ:
                raise AuthenticationException(
                    "UNITY_API_KEY not found"
                )
            self.api_key = os.environ[UNITY_API_KEY]
        # Override if provided as an argument
        self.api_key = self.api_key

        self.endpoint = endpoint

        self._auth = {
            "X-AUTH-TOKEN": self.api_key
        }
        self._headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            "User-Agent": "",
            'X-User-Agent': f'unity-vision-sdk {_SDK_VERSION})',
        }

    def _get(self,
              uri,
              query=None,
              params=None,
              timeout=30.0,
              filters=None,
              *args,
              **kwargs):
        """"""
        data = None
        if query is not None:
            data = json.dumps({
                'query': query,
                'variables': params
            }).encode("utf-8")

        try:
            request = {
                'uri': uri,
                'data': data,
                'headers': self._headers,
                'timeout': timeout
            }

            response = request.get(**request)
        except requests.exceptions.Timeout as e:
            raise TimeoutException(str(e))
        except Exception as e:
            raise UnityVisionException(
                "Unknown error during Client.get(): " + str(e), e)

        try:
            res_json = response.json()
        except Exception as e:
            raise UnityVisionException(f"Failed to parse response JSON: {str(e)}")

        if "error" in res_json:
            raise UnityVisionException(f"Error in response: {res_json['error']}")

        return res_json["data"]

    def _download_from_signed_url(self, signed_uri):
        """"""
        return requests.get(signed_uri)

    def _get_ucvd_dataset(self, run_id):
        """"""
        __entity_uri = Path(self.endpoint, "datasets")
        ucvd_dataset_signed_uri = self._get(
            uri=__entity_uri,
            query=run_id,
            params=None,
        )

        return self._download_from_signed_url(ucvd_dataset_signed_uri)
