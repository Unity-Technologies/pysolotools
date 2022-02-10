from abc import ABC, abstractmethod


class DatasetClient(ABC):
    @abstractmethod
    def create_dataset(self, cfg):
        pass

    @abstractmethod
    def download_dataset(self, run_id):
        pass

    @abstractmethod
    def describe_dataset(self, run_id):
        pass

    @abstractmethod
    def list_datasets(self):
        pass
