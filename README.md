# Table of Contents

1. [Introduction](#introduction)
    * [Basic Framework](#intro_basic)
    * [Reward Shaping Extension](#intro_shaping)
    

2. [Quick Start](#quickstart)
    * [Installation](#installation)
    * [Running](#running)


3. [Experiments](#experiments)
    * [Hyperparameter-Optimization](#hyperparameter)
    * [Conservative Agent](#conservative)

# Introduction <a name="introduction"></a>
Hanabi Multiagent Framework (HMF) provides a link between the Hanabi Learning Environment and different types of agents, for example rule based or rainbow agents. Learning agents can be trained in self-play mode or in a multi-agent setting.

## Basic Framework <a name="intro_basic"></a>
## Reward Shaping Extension <a name="intro_shaping"></a>

# Quick Start <a name="quickstart"></a>

## Installation <a name="installation"></a>
Make sure that all requirements are satisfied before installing the ```hanabi-multiagent-framework``` package. Install from git:
```
$ pip install git+https://github.com/Hanabi-Game-Project/hanabi-multiagent-framework.git@conservative_agent
```
### Requirements
The following packages need to be installed:
* wheel
* cmake
* numpy
* gin-config
* chex==0.0.2
* dm-env==1.3
* dm-haiku==0.0.3
* dm-tree==0.1.5
* jax==0.2.5
* jaxlib for GPU (```https://storage.googleapis.com/jax-releases/cuda100/jaxlib-0.1.55-cp36-none-manylinux2010_x86_64.whl```)
* optax==0.0.1
* rlax==0.0.2
* ```https://github.com/Hanabi-Game-Project/hanabi-learning-environment.git@pybind11```
* ```https://github.com/Hanabi-Game-Project/hanabi-agents.git@conservative_agent```
* ```https://github.com/vsois/hanabi-agents.git@level_shaping#subdirectory=sum_tree```

### Docker
For convenience a docker file is provided: `docker/images/Dockerfile-gpu-rlax`

To build a docker container with GPU support and rlax installed, run:
```
$ cd docker/images
$ sudo docker build -t hanabi-framework:gpu-rlax -f Dockerfile-gpu-rlax .
```
Within the container, all necessary components are installed (see requirements). To run the container, execute:
```
$ sudo docker run -it --gpus all -v /path_to_repo/hanabi-multiagent-framework/examples:/examples:ro hanabi-framework:gpu-rlax bash
```
The docker reference provides further information about [building](https://docs.docker.com/engine/reference/commandline/build/) and [running](https://docs.docker.com/engine/reference/commandline/container_run/) a docker container.

## Running <a name="running"></a>
In the `examples` folder a python script `reward_shaping_session.py` and a configuration file `rlax_agent.gin` for training a rainbow agent are provided. The configuration file contains the agent parameters determined by Hyperparameter-Optimization, described in the chapter below.

Run the following command (in the docker container). The weights and statistics generated during training are stored in the folder `/output_example`.
```
$ python /examples/reward_shaping_session.py --agent_config_path=/examples/rlax_agent.gin --output_dir=/output_example
```
To train an agent in self-play mode, run the script with the `self_play` argument:
```
$ python /examples/reward_shaping_session.py --self_play
```

Pretrained weights of training and target network, the optimizer state and the experience buffer can be loaded for each agent. The information must be given in json-format, for example:
```
$ python /examples/reward_shaping_session.py --self_play --agent_config_path=/examples/rlax_agent.gin --start_with_weights="{\"agent_0\" : [\"/examples/weights/conservative/rlax_rainbow_best_online.pkl\",\"/examples/weights/conservative/rlax_rainbow_best_target.pkl\"]}"
```
If using pretrained weights, the network structure must remain unchanged. Loading the experience buffer from a file overwrites buffer parameters set in the gin file. To continue a training session without information loss, the files containing network parameters, optimizer state and experience buffer must be loaded into the session. Adjust the ```epoch_offset``` in the gin file to reflect the number of epochs trained previously.

Further parameters of the training session can be adjusted in the gin configuration file:

| Parameter | Description | Default |
|:---|:---|:---|
| hanabi_game_type | Game configuration, can be:<br />Hanabi-{VerySmall,Small,Full}-{Oracle,CardKnowledge} | Hanabi-Small |
| n_players | Number of players | 2 |
| max_life_tokens | Adjusted number of life tokens | None |
| n_parallel | Number of games run in parallel during training | 32 |
| n_parallel_eval | Number of games run in parallel during evaluation  | 1000 |
| n_train_steps | Number of training steps made in each iteration. <br />One iteration consists of n_sim_steps followed by n_train_steps. | 4 |
| n_sim_steps | Number of environment steps made in each iteration | 2 |
| epochs | Number of epochs <br />One epoch consists of eval_freq iterations for each player | 1,000,000 |
| epoch_offset | Number of epochs run in a previous training session<br /> Use when loading pretrained weights to continue training | 0 |
| eval_freq | Number of iterations to perform between evaluations | 500 |
| n_backup | Number of epochs to run before creating a backup of the agents' state | 500 |

# Experiments <a name="experiments"></a>

## Hyperparameter-Optimization <a name="hyperparameter"></a>
## Conservative Agent  <a name="conservative"></a>
