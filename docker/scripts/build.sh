#!/bin/bash

DOCKERFILE_DIR="$( cd "$( dirname $(dirname "${BASH_SOURCE[0]}") )" >/dev/null 2>&1 && pwd )/images"

HELP_STRING="Build a docker image with hanabi framework.\n
             You have to specify one TAG. supported TAGs are:\n
             \t--cpu-pytorch-1.4.0\n
             \t--cpu-tf-1.5.2\n
             \t--tf-2.1.0."

case "$1" in
  --cpu-pytorch-1.4.0)
    echo "Building cpu-pytorch image flavor."
    TAG="cpu-pytorch-1.4.0"
    ;;
  --cpu-tf-1.5.2)
    echo "Building cpu-tf-1.5.2 image flavor."
    TAG="cpu-tf-1.5.2"
    ;;
  --tf-2.1.0)
    echo "Building tf-2.1.0 image flavor."
    TAG="tf-2.1.0"
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
    echo "unknown option $1. Abort."
    exit 1
    ;;
esac

docker build -t hanabi-env:latest -f $DOCKERFILE_DIR/Dockerfile-hanabi-env $DOCKERFILE_DIR
docker build -t hanabi-framework:$TAG -f $DOCKERFILE_DIR/Dockerfile-$TAG $DOCKERFILE_DIR

echo ""
echo "Done building image <hanabi-framework:$TAG>."
