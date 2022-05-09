import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    createdAt: datetime.date = None
    updatedAt: datetime.date = None
    description: str = None
    licenseURI: str = None


@dataclass(frozen=True)
class Archive(object):
    id: str
    name: str
    type: str
    state: dict = None
    downloadURL: str = None
    uploadURL: str = None
    createdAt: datetime.date = None
    updatedAt: datetime.date = None


@dataclass(frozen=True)
class Attachment(object):
    id: str
    name: str
    description: str = None
    state: dict = None
    downloadURL: str = None
    uploadURL: str = None
    createdAt: datetime.date = None
    updatedAt: datetime.date = None
