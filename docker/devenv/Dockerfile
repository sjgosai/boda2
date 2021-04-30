# Available images: 
# $ gcloud container images list --repository="gcr.io/deeplearning-platform-release"
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-7

ENV SHELL=/bin/bash

WORKDIR /usr/src

# Container doesn't have UNZIP
RUN apt-get update
RUN apt-get install unzip

# Dumping contents of repo into container
ADD . /usr/src/boda2/
WORKDIR /usr/src/boda2/

# Installing requirements, including shipped submodules
RUN pip install --upgrade pip==20.2.3
RUN pip install --no-cache-dir --use-feature=2020-resolver -r requirements.txt

# Installing BODA as a package
RUN pip install -e .

WORKDIR /
