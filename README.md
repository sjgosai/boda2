# boda2
Bayesian Optimization of DNA Activity (BODA) version 2

# Get directory permissions

```
sudo chown -R $USER /home/$USER
```

# Authenticate Docker

```
sudo usermod -a -G docker ${USER}
gcloud auth login
gcloud auth configure-docker
bash src/run_docker_for_dev.sh gcr.io/sabeti-encode/boda devenv 0.0.1
```
