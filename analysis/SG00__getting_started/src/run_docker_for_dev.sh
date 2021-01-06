%%sh

REPO=${1:-"headjack"}
IMAGE_TAG=${2:-"latest"}
for_cpu=${3:-false}

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}

PORT=8888
container=${account}.dkr.ecr.${region}.amazonaws.com/$REPO:$IMAGE_TAG
cmd_on_start="jupyter notebook --ip 0.0.0.0 --port ${PORT} --no-browser --allow-root"


# Get the login command from ECR and execute it directly
$(aws ecr get-login --region us-east-1 --no-include-email)

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 763104351884 --region us-east-1 --no-include-email)

echo "Pulling container ${container}"
echo "docker run --gpus all -it --shm-size=64g -p ${PORT}:${PORT} -p 6006:6006 -v /home/${USER}:/home/${USER} $container $cmd_on_start"
echo ${cmd_on_start}

if $for_cpu
then
    gpu_flag=""
else
    gpu_flag='--gpus all'
fi

docker run $gpu_flag -it --shm-size=64g -p $PORT:$PORT -p 6006:6006 -v /home/${USER}:/home/${USER} -v ~/.aws:/root/.aws $container $cmd_on_start
