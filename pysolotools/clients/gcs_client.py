import os

from google.cloud import exceptions
from google.cloud.storage import Client

from pysolotools.core.exceptions import DatasetNotFoundException


class GCSClient:
    def __init__(self, client=Client()):
        self.client = client

    def download_directory(self, uri, dest_path, verbose=False) -> bool:
        """
        Downloads a directory from GCS.

        Args:
            uri (str): GCS path to parent directory containing SOLO dataset
            dest_path: Path to copy the blobs into
            verbose:   Verbosity for logs

        Returns:
            bool: Success or failure flag
        """
        bucket_name = uri.split("/")[2]
        prefix = "/".join(uri.split("/")[3:])
        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                file_rel_to_source = os.path.relpath(blob.name, prefix)
                full_path = os.path.join(dest_path, file_rel_to_source)
                dir_name = os.path.dirname(full_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                blob.download_to_filename(full_path)
            if verbose:
                print(f"Directory {prefix} downloaded to {dest_path}")
            return True
        except exceptions.NotFound as ne:
            raise DatasetNotFoundException(message="Dataset not found", source=ne)
        except exceptions.ClientError:
            return False
