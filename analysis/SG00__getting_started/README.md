# Getting started

In this directory we are developing a custom Docker container that we use for environment consistancy, and usage with the Google AI Platform. The dummy model here is a simple CIFAR10 implementation using PyTorch Lightning.

## Build and Push a Docker container

This will build a container with the URI: `gcr.io/sabeti-encode/cifar:0.0.0`. Refer to `Dockerfile` for specifics about how the container is built.

```
bash src/build_and_push.sh gcr.io/sabeti-encode cifar 0.0.0 Dockerfile
```

It's important to be aware that the `ENTRYPOINT` for the container is set to `["python", "/opt/ml/code/main.py"]`, so simply attempting to mount the container will invoke training code unless the `docker run` command specifies and alternative `--entrypoint`.


## Run Jupyter in a container

We can connect to the container using another helper script that takes care of a few particulars.

```
bash src/run_docker_for_dev.sh gcr.io/sabeti-encode cifar 0.0.0 8889 6007
```

This call will mount a container from `gcr.io/sabeti-encode/cifar:0.0.0` and connect two ports (e.g., 8889 and 6007) within the container to the same two ports outside of the container. This is helpful if, say, you tunnel into your VMs with something like:

```
ssh -i ~/.ssh/my_gcp_key -L 8889:localhost:8889 username@ip.address.here
```

Then you can follow a port all the way from your local machine, to the remote container.

## Interactive model testing

Use `run_docker_for_dev.sh` and use Jupyter Notebooks and Terminals to test code. This example here works with:

```
python /opt/ml/code/main.py \
  --model_dir=/tmp/pytorch-example/model \
  --data_dir=/tmp/pytorch-example/data \
  --gpus=1 \
  --max_epochs=1
```

Can also tune a trained model:

```
python /opt/ml/code/main.py \
  --model_dir=/tmp/transfer-example/model \
  --pretrained_model=/tmp/pytorch-example/model
  --data_dir=/tmp/pytorch-example/data \
  --gpus=1 \
  --max_epochs=1
```

## Google AI Platform (Classic)

We've also got this basic thing working on Google AI Platform. We can submit a single training job as follows:

```
export BUCKET_NAME=syrgoth
export MODEL_DIR=cifar_model_$(date +%Y%m%d_%H%M%S)
export REGION=us-central1
export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
  --master-machine-type n1-standard-8 \
  --master-accelerator count=1,type=nvidia-tesla-v100 \
  --scale-tier CUSTOM \
  --region $REGION \
  --master-image-uri gcr.io/sabeti-encode/cifar:0.0.0 \
  -- \
  --model_dir=gs://$BUCKET_NAME/$MODEL_DIR \
  --data_dir=/tmp/pytorch-example/ \
  --gpus=1 \
  --max_epochs=1
```

Note, before the `-- ` (empty???) arg you put all of the GAIP specific args, and after you put our code specific args.

## Hyperparameter Tuning

In progress

## Google AI Platform (Unified)

In November 2020, Google released the "Unified" version of GAIP. Haven't tried it yet. The docs look better organized.

## References and notes for later:

Google AI Platform [Classic](https://cloud.google.com/ai-platform/docs) and [Unified](https://cloud.google.com/ai-platform-unified/docs) docs

[Training SDK](https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training)

Custom container [Prediction](https://cloud.google.com/ai-platform/prediction/docs/getting-started-pytorch-container)

[Hyperparameter Tuning](https://cloud.google.com/ai-platform/training/docs/hyperparameter-tuning-overview)