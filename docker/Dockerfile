FROM python:3.10-slim-bullseye

RUN apt update && apt install ffmpeg libsm6 libxext6 -y

WORKDIR /workspace/
ADD requirements/ /workspace/requirements/
ADD README.md /workspace/README.md
ADD setup.py /workspace/setup.py
ADD pysolotools/ /workspace/pysolotools/

RUN pip install .