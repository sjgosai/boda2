# Available images: 
# $ gcloud container images list --repository="gcr.io/deeplearning-platform-release"
FROM gcr.io/sabeti-encode/boda/devenv:0.1.1

# Installing BODA as a package
ADD . /opt/boda2/
WORKDIR /opt/boda2/
RUN pip install -e .

WORKDIR /
