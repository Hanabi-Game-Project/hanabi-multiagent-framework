# Introduction

Hanabi Multiagent Framework (HMF) is a convenience package automatize running a game of hanabi with several parallel states with multiple RL agents.

# Installation

Dependencies:
 * numpy
 * [dm_env](https://github.com/deepmind/dm_env)
 * modified version of [hanabi-learning-environment](https://github.com/braintimeException/hanabi-learning-environment)

For convenience we provide a docker file where most of the dependencies (including those for running compatible agents, see [hanabi-agents](https://github.com/braintimeException/hanabi-agents))

For example, to build a docker container with GPU support and rlax installed, run 
```
# clone this repo if you haven't yet
$ git clone https://github.com/braintimeException/hanabi-multiagent-framework/ --branch pybind-env

# build the container
$ cd docker/images
$ docker build -t hanabi-framework:gpu-rlax -f Dockerfile-gpu-rlax .
```

# Running

The docker container contains all necessary dependencies along with the
`hanabi-agents` repo. In order to try it out, run following commands:
```
# run the container with a bash session
$ docker run -it --rm --gpus=all \
    --volume /path_to_repo/hanabi-multiagent-framework/examples:/hanabi-examples:ro \
    hanabi-framework:gpu-rlax bash

# try out examples
# e.g. train a rlax rainbow agent using a provided config
python hanabi-examples/rlax_agent_session.py --agent_config_path=hanabi-framework/rlax_agent.gin
# or let rule-based agent play with itself
python hanabi-examples/rulebased_agent_session.py
```
