#!/usr/bin/env python
"""
Installs pysolo
"""

import io
import os
from os.path import dirname, realpath

from setuptools import setup

# Package meta-data.
NAME = "pysolo"
DESCRIPTION = "unity computer vision dataset toolchain"
URL = "https://https://github.com/Unity-Technologies/pysolo"
EMAIL = "souranil@unity3d.com"
AUTHOR = "Unity Technologies"
REQUIRES_PYTHON = ">=3.7"
VERSION = "0.3.0"


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _read_requirements():
    requirements = f"{dirname(realpath(__file__))}/requirements.txt"
    with open(requirements) as f:
        results = []
        for line in f:
            line = line.strip()
            if '-i' not in line:
                results.append(line)
        return results


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=[NAME],
    include_package_data=True,
    license="MIT",
    install_requires=[
        "requests~=2.25.1",
        "requests-toolbelt>=0.9.1",
        "tqdm~=4.62.3",
        "responses~=0.20.0",
        "ratelimit~=2.2.1",
        "dataclasses-json==0.5.7"
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=[
        "pysolo",
        "pysolo.core",
        "pysolo.core.iterators",
        "pysolo.consumers",
        "pysolo.clients"
    ]
)
