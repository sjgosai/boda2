# NVIDIA Docker Container Tool-kit

## Check if installed
If this runs, you're good to go.
```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

If this failed, move on to steps below. See References below for helpful links.

## Uninstall previous versions of Docker

```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

# Setup instructions

## Install Docker CE with convenience scripts

```
curl https://get.docker.com | sh \
  && sudo systemctl start docker \
  && sudo systemctl enable docker
```

## Set up NVIDIA container tool-kit

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-docker2
```

```
sudo systemctl restart docker
```

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Configure docker

```
sudo groupadd docker
```

```
sudo usermod -aG docker $USER
```

```
newgrp docker
```

```
docker run hello-world
```

## If access to `~/.docker/*` config files denied

```
sudo chown "$USER":"$USER" /home/"$USER"/.docker -R
sudo chmod g+rwx "$HOME/.docker" -R
```

# References

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

https://docs.docker.com/engine/install/ubuntu/

https://docs.docker.com/engine/install/linux-postinstall/
