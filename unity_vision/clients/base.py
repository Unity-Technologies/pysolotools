
from abc import ABC, abstractmethod


class DatasetClient(ABC):
    @abstractmethod
    def create_dataset(self, cfg):
        pass

    @abstractmethod
    def download_dataset_archives(self, dataset_id: str, dest_dir: str, **kwargs):
        pass

    @abstractmethod
    def describe_dataset(self, dataset_id: str):
        pass

    @abstractmethod
    def list_datasets(self):
        pass
