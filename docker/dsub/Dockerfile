# Available images: 
# $ gcloud container images list --repository="gcr.io/deeplearning-platform-release"
FROM gcr.io/sabeti-encode/boda/devenv:0.1.1

# Container doesn't have less
#RUN apt-get update
#RUN apt-get install less

# Installing BODA as a package
ADD . /home/ubuntu/boda2/
WORKDIR /home/ubuntu/boda2/
RUN pip install -e .

ADD . /opt/boda2/

WORKDIR /
