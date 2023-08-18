# Development Environment Dockerfile
# Available images: 
# $ gcloud container images list --repository="gcr.io/deeplearning-platform-release"
FROM gcr.io/sabeti-encode/boda/base:0.2.0

# Installing BODA as a package
# installing the version in home
# that well edit during development
ADD . /home/ubuntu/boda2/
WORKDIR /home/ubuntu/boda2/
RUN pip install -e .

# Go to root
WORKDIR /
