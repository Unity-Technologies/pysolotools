import glob
import logging
import os
import tarfile

import requests
import requests.exceptions
import requests_toolbelt
from ratelimit import limits
from requests.auth import HTTPBasicAuth

from pysolotools.core.exceptions import AuthenticationException, UCVDException
from pysolotools.core.models import UCVDArchive, UCVDAttachment, UCVDDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

BASE_URI_V1 = "https://services.api.unity.com/computer-vision-datasets/v1"
UNITY_AUTH_SA_KEY = "UNITY_AUTH_SA_KEY"
UNITY_AUTH_API_SECRET = "UNITY_AUTH_API_SECRET"
_SDK_VERSION = "v0.0.1"


class UCVDClient:
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
        rate_limit_period=900,
        **kwargs,
    ):
        """
        Creates and initializes a UCVDClient

        Usage:

        >> client = UCVDClient(
            org_id="<unity-org-id>",
            project_id="<unity-project-id>",
            sa_key="UNITY_AUTH_SA_KEY",
            api_secret="UNITY_AUTH_API_SECRET"
            )

        Args:
            org_id (str): Organization ID
            project_id (str): Project ID
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
        self.endpoint = (
            f"{base_uri}/organizations/{self.org_id}/projects/{self.project_id}"
        )
        self.api_version = api_version
        self.auth = HTTPBasicAuth(self.sa_key, self.api_secret)
        self.rate_limit_period = rate_limit_period
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "",
            "X-User-Agent": f"unity-vision-sdk {_SDK_VERSION})",
        }

    def create_dataset(self, dataset_name, description=None, license_uri=None):
        entity_uri = f"{self.endpoint}/datasets"
        dataset = {"name": dataset_name}
        if description:
            dataset["description"] = description
        if license_uri:
            dataset["licenseURI"] = license_uri
        return self.__make_request(
            method="post", url=entity_uri, body=dataset, auth=self.auth
        )

    def describe_dataset(self, dataset_id):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload

    def list_datasets(self):
        entity_uri = f"{self.endpoint}/datasets"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload["results"]

    def list_dataset_archives(self, dataset_id):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/archives"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload["results"]

    def create_dataset_archive(self, dataset_id, archive_name="Archive.tar"):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/archives"
        body = {"name": archive_name}
        return self.__make_request(
            method="post", url=entity_uri, body=body, auth=self.auth
        )

    def describe_dataset_archive(self, dataset_id, archive_id):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/archives/{archive_id}"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload

    def list_dataset_attachments(self, dataset_id):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/attachments"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload["results"]

    def describe_dataset_attachment(self, dataset_id, attachment_id):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/archives/{attachment_id}"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload

    def create_dataset_attachment(self, dataset_id, attachment_name, description):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/attachment"
        body = {"name": attachment_name, "description": description}
        return self.__make_request(
            method="post", url=entity_uri, body=body, auth=self.auth
        )

    def create_build(self, build_name, description):
        entity_uri = f"{self.endpoint}/builds"
        body = {"name": build_name, "description": description}
        return self.__make_request(
            method="post", url=entity_uri, body=body, auth=self.auth
        )

    def list_builds(self):
        entity_uri = f"{self.endpoint}/builds"
        payload = self.__make_request(method="get", url=entity_uri, auth=self.auth)
        return payload["results"]

    def describe_build(self, build_id: str):
        entity_uri = f"{self.endpoint}/builds/{build_id}"
        return self.__make_request(method="get", url=entity_uri, auth=self.auth)

    @limits(calls=15, period=900)
    def iterate_datasets(self):
        entity_uri = f"{self.endpoint}/datasets"
        for res in self.__iterable_get(entity_uri, auth=self.auth):
            yield UCVDDataset(**res)

    @limits(calls=15, period=900)
    def iterate_dataset_archives(self, dataset_id):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/archives"
        for res in self.__iterable_get(entity_uri, auth=self.auth):
            yield UCVDArchive(**res)

    @limits(calls=15, period=900)
    def iterate_dataset_attachments(self, dataset_id):
        entity_uri = f"{self.endpoint}/datasets/{dataset_id}/attachments"
        for res in self.__iterable_get(entity_uri, auth=self.auth):
            yield UCVDAttachment(**res)

    @staticmethod
    def extract_dataset(solo_dir, subdir=False):
        if subdir:
            root_path = f"{solo_dir}/solo"
        else:
            root_path = solo_dir

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        for archive in glob.glob(f"{solo_dir}/*.tar"):
            print(f"Extracting {archive} to {root_path}")
            with tarfile.open(archive) as tar:
                tar.extractall(root_path)

    def download_dataset_archives(
        self,
        dataset_id: str,
        dest_dir: str,
        chunk_size=1024**2,
        skip_on_error: bool = True,
    ):
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        for archive in self.iterate_dataset_archives(dataset_id):
            if archive.state["status"] != "READY":
                continue
            r = requests.get(archive.downloadURL, stream=True)
            if not r.ok:
                if skip_on_error:
                    continue
                else:
                    r.raise_for_status()
            with open(os.path.join(dest_dir, archive.name), "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        logger.info("Downloaded successfully")

    def download_dataset_attachment(
        self,
        dataset_id: str,
        attachment_id: str,
        dest_dir: str = ".",
        chunk_size: int = 1024**2,
    ):
        attachment = self.describe_dataset_attachment(dataset_id, attachment_id)
        if attachment.state["status"] == "READY":
            r = requests.get(attachment.downloadURL, stream=True)
            r.raise_for_status()
            with open(os.path.join(dest_dir, attachment.name), "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
            return
        raise Exception(f"attachment {attachment_id} is not ready for download")

    @staticmethod
    def upload_file(url: str, filename: str) -> requests.Response:
        size = os.path.getsize(filename)
        with open(filename, "rb") as fd:
            stream = requests_toolbelt.StreamingIterator(size, fd)
            return requests.put(url, data=stream)

    @staticmethod
    def __iterable_get(base_url, headers=None, auth=None, params=None):
        url = base_url
        params = params or {}
        while True:
            res = requests.get(url, params=params, headers=headers, auth=auth)
            res.raise_for_status()
            payload = res.json()
            for result in payload["results"]:
                yield result
            if "next" not in payload:
                break
            url = f"{base_url}?next={payload['next']}"

    @staticmethod
    def __make_request(
        method, url, headers=None, auth=None, params=None, body=None, data=None
    ):
        params = params or {}
        session = requests.Session()
        try:
            res = session.request(
                method=method,
                url=url,
                headers=headers,
                auth=auth,
                params=params,
                json=body,
                data=data,
            )
            res.raise_for_status()
            session.close()
            return res.json()
        except Exception as err:
            session.close()
            logger.error(str(err))
            raise UCVDException(str(err))
