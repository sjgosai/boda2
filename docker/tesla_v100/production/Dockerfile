# GCP Vertex AI platform Dockerfile
# Available images: 
# $ gcloud container images list --repository="gcr.io/deeplearning-platform-release"
FROM gcr.io/sabeti-encode/boda/base:0.1.0

# Installing BODA as a package
# installing the a copied version
# away from home where we edit
ADD . /opt/boda2/
WORKDIR /opt/boda2/
RUN pip install -e .

# Transfering source code for model training, etc.
ENV PATH="/opt/ml/code:${PATH}"
ADD src /opt/ml/code

WORKDIR /

ENTRYPOINT ["python", "/opt/ml/code/main.py"]
