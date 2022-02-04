from google.cloud.storage import Client


class GCSClient:
    def __init__(self, **kwargs):
        self.client = Client(**kwargs)
