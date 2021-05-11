#!/bin/bash
  
docker_registry=$1
container_name=$2
container_version=$3
dockerfile=${4:-Dockerfile}

echo "Registering container to:"
fullname="${docker_registry}/${container_name}:${container_version}"
echo "${fullname}"

docker build -t ${container_name} -f ${dockerfile} .
docker tag ${container_name} ${fullname}

docker push ${fullname}