import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    createdAt: datetime.date = None
    updatedAt: datetime.date = None
    description: str = None
    licenseURL: str = None
