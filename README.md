<div align="center">
<p align="center">


**The open-source tool for using Unity SOLO Datasets**
---

[![PyPI version](https://github.com/pytest-dev/pytest-cov/actions/workflows/test.yml/badge.svg)](https://github.com/Unity-Technologies/pysolotools/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</p>
</div>

## Introduction

`pysolotools` would allow you to the following:

- Parse SOLO datasets generated with Unity Perception.
- Dataset iterables.
- Interface for nodes.
- Clients to access remote datasets
  - Unity Computer Vision Datasets (UCVD)

You can read more about SOLO schema [here](https://github.com/Unity-Technologies/perception/blob/main/com.unity.perception/com.unity.perception/Documentation~/SoloSchema/Solo_Schema.md).

## Pre-Requisites
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended). Install [pre-commit](https://pre-commit.com/).
- Make sure `pip` is installed.

## Installation

---
To install, run
```shell
pip install pysolotools --index-url=https://artifactory.prd.it.unity3d.com/artifactory/api/pypi/pypi/simple
```

** The package lives in the internal PyPi repo for now.



#### SOLO Dataset

##### Iterating Frames

```python
from pysolotools.consumers import Solo
solo = Solo(data_path="<<dataset-path>>")

for frame in solo.frames():
    frame
```

Refer to `pysolotools.core.iterator.FrameIterator`

##### Cloud Datasets

* [Datasets from UCVD APIs](docs/UCVD.md)
* [GCS](docs/GCS.md)

## Sphinx Docs

[Github Pages](https://effective-train-86190335.pages.github.io/)

To generate docs locally:

1. `cd docs/`
2. `make html`

If you want to rebuild the rst files, please run: `make apidoc`. This will generate the rst files based off docstring comments. (Note: This does not update existing rst files).


## Additional Resources

### Blog Posts and Talks

- Data-centric AI with Unity Computer Vision Datasets [blogpost](https://blog.unity.com/technology/data-centric-ai-with-unity-computer-vision-datasets)
- Workshop notebook [Notebook](https://colab.research.google.com/drive/1yoR-47aGi9L0_3f0ULq9Udk0cC64V-0-?usp=sharing)


## Community and Feedback

The Unity Computer Vision demos are open-source and we encourage and welcome contributions.
If you wish to contribute, be sure to review our [contribution guidelines](CONTRIBUTING.md)
and [code of conduct](CODE_OF_CONDUCT.md).

## Support

For feature requests, bugs, or other issues, please file a
[GitHub issue](https://github.com/Unity-Technologies/Unity-Vision-Hub/issues)
using the provided templates we will investigate as soon as possible.

## Newsletter

## License
[Apache License 2.0](LICENSE)
