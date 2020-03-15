#!/bin/bash

DOCKERFILE_DIR="$( cd "$( dirname $(dirname "${BASH_SOURCE[0]}") )" >/dev/null 2>&1 && pwd )/images"

HELP_STRING="Build a docker image with hanabi framework.\n
             You have to specify one TAG. supported TAGs are:\n
             \t--cpu-pytorch-1.4.0\n
             \t--cpu-tf-1.15.2\n
             \t--tf-2.1.0."

case "$1" in
  --cpu-pytorch-1.4.0)
    echo "Building cpu-pytorch image flavor."
    ENV_DOCKERFILE="Dockerfile-hanabi-env"
    HW_TAG="cpu"
    TAG="pytorch-1.4.0"
    ;;
  --cpu-tf-1.15.2)
    echo "Building cpu-tf-1.15.2 image flavor."
    ENV_DOCKERFILE="Dockerfile-hanabi-env"
    TAG="tf-1.15.2"
    HW_TAG="cpu"
    ;;
  --gpu-tf-1.15.2)
    echo "Building gpu-tf-1.15.2 image flavor."
    ENV_DOCKERFILE="Dockerfile-hanabi-env"
    TAG="tf-1.15.2"
    HW_TAG="gpu"
    ;;
  --tf-2.1.0)
    echo "Building tf-2.1.0 image flavor."
    ENV_DOCKERFILE="Dockerfile-hanabi-env"
    TAG="tf-2.1.0"
    HW_TAG="gpu"
    ;;
  -h)
    echo -e $HELP_STRING
    exit 0
    ;;
  --help)
    echo -e $HELP_STRING
    exit 0
    ;;
  *)
    echo "[ERROR] Unknown option $1. Abort."
    exit 1
    ;;
esac

docker build -t hanabi-env:$HW_TAG -f $DOCKERFILE_DIR/$ENV_DOCKERFILE-$HW_TAG $DOCKERFILE_DIR
docker build -t hanabi-framework:$HW_TAG-$TAG -f $DOCKERFILE_DIR/Dockerfile-$HW_TAG-$TAG $DOCKERFILE_DIR

echo ""
echo "Done building image <hanabi-framework:$HW_TAG-$TAG>."
