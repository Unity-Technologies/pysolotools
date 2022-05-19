"""
UCVD API Dataclasses
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class UCVDDataset:
    id: str
    name: str
    createdAt: datetime.date = None
    updatedAt: datetime.date = None
    description: str = None
    licenseURI: str = None


@dataclass(frozen=True)
class UCVDArchive(object):
    id: str
    name: str
    type: str
    state: dict = None
    downloadURL: str = None
    uploadURL: str = None
    createdAt: datetime.date = None
    updatedAt: datetime.date = None


@dataclass(frozen=True)
class UCVDAttachment(object):
    id: str
    name: str
    description: str = None
    state: dict = None
    downloadURL: str = None
    uploadURL: str = None
    createdAt: datetime.date = None
    updatedAt: datetime.date = None
