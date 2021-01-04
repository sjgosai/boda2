# Available images: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04

ENV SHELL=/bin/bash

WORKDIR /usr/src

# Container doesn't have UNZIP
RUN apt-get update
RUN apt-get install unzip

# Dumping contents of repo into container
ADD . /usr/src/cifar10/
WORKDIR /usr/src/cifar10/

# Installing requirements, including shipped submodules
RUN pip install --upgrade pip==20.2.3
RUN pip install --no-cache-dir --use-feature=2020-resolver -r requirements.txt

# Installing Headjack
RUN pip install -e .

# Transfering source code for model training, etc.
ENV PATH="/opt/ml/code:${PATH}"
ADD src /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM main.py

WORKDIR /
