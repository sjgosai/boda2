#!/bin/bash

docker_registry=$1
container_name=$2
container_version=$3
PORT_A=${4:-8888}
PORT_B=${5:-6006}
for_cpu=${6:-false}

container=${docker_registry}/${container_name}:${container_version}

cmd_on_start="jupyter notebook --ip 0.0.0.0 --port ${PORT_A} --no-browser --allow-root"

if $for_cpu
then
    gpu_flag=""
else
    gpu_flag='--gpus all'
fi

docker run $gpu_flag -it --shm-size=64g -p ${PORT_A}:${PORT_A} -p ${PORT_B}:${PORT_B} -v /home/${USER}:/home/${USER} -v ~/.gsutil:/root/.gsutil $container $cmd_on_start
