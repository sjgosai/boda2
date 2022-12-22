#!/bin/bash
  
container_version=$1

# install git
sudo apt-get update
sudo apt-get install git

# install docker
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
    
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# make docker available with general user privilege
sudo gpasswd -a $(whoami) docker

# docker build, tag and push
git clone https://github.com/sjgosai/boda2.git
cd boda2
docker build -f docker/dsub/Dockerfile -t dsub . --no-cache
fullname="$gcr.io/sabeti-encode/dsub:${container_version}"
docker tag dsub ${fullname}
docker push ${fullname}