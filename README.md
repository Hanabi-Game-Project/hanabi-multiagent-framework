# Introduction

Hanabi Multiagent Framework (HMF) is a convenience package automatize running a game of hanabi with several parallel states with multiple RL agents.

# Installation

Dependencies:
 * numpy
 * [dm_env](https://github.com/deepmind/dm_env)
 * modified version of [hanabi-learning-environment](https://github.com/braintimeException/hanabi-learning-environment)

For convenience we provide a docker file where most of the dependencies (including those for running rlax-based RL agents, see [hanabi-agents](https://github.com/braintimeException/hanabi-agents))

For example, to build a docker container with GPU support and rlax installed, run 
```
# clone this repo if you haven't yet
$ git clone https://github.com/braintimeException/hanabi-multiagent-framework/

# build the container
$ cd docker/images
$ docker build -t hanabi-framework:gpu-rlax -f Dockerfile-gpu-rlax .
```

# Running

The docker container above is a development version, meaning that it does not contain code neither from this repo, nor from `hanabi-learning-environment`, nor from `hanabi-agents`. Therefore, to run the code you would need to clone these repos and mount them, like so:
```
# clone the repos
$ git clone https://github.com/braintimeException/hanabi-learning-environment --branch feature/parallel-env
$ git clone https://github.com/braintimeException/hanabi-agents

# run the container with a bash session
$ docker run -it --gpus=all \
    --volume /path_to_repo/hanabi-multiagent-framework:/hanabi-framework:ro \
    --volume /path_to_repo/hanabi-learning-environment:/hanabi-le:ro \
    --volume /path_to_repo/hanabi-agents/:/hanabi-agents:ro \
    hanabi-framework:gpu-rlax bash
```

This inconvenience is due to active development stage of the package. We are going to provide a version of the container with all dependencies later.

From within the container you should install the repos:

```
$ pip install /hanabi-framework/
$ pip install /hanabi-learning-environment/
$ pip install /hanabi-agents/
```

There are some examples showing how to run the framework. For instance, the rlax_agent_session.py shows how to run the framework with a rlax-based DQN agent awailable in hanabi_agents repo. You can launch like so:

```
$ python /hanabi-framework/examples/rlax_agent_session.py
```
