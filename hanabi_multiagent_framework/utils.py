"""
A collection of helper functions
"""
from hanabi_learning_environment import pyhanabi, rl_env

def canonical_obsevation_substitute_colors(vectorized_observation, substitution_map, offsets):
    """Replace colors in the observation according to the provided map
    """
    # EncodeHands, EncodeBoard, EncodeDiscards, EncodeLastAction, (EncodeCardKnowledge)
    raise NotImplementedError()


def make_hanabi_env_config(
        environment_name="Hanabi-Full", num_players=2):
    """Generate a configuration dictionary for hanabi environment.

    Args:
        environment_name (str) -- name of the environment to generate, one of the following:
                                   - 'Hanabi-Full' or 'Hanabi-Full-CardKnowledge': use 5 colors, 5
                                     ranks, 8 info tokens, 3 life tokens and default card knowledge.
                                   - 'Hanabi-Full-Minimal': use 5 colors, 5
                                     ranks, 8 info tokens, 3 life tokens and minimal card knowledge.
                                   - 'Hanabi-Small': use 2 colors, 5
                                     ranks, 3 info tokens, 1 life token and default card knowledge.
                                   - 'Hanabi-Very-Small': use 1 colors, 5
                                     ranks, 3 info tokens, 1 life token and default card knowledge.
                                   - 'Hanabi-Very-Small-Oracle': use 1 colors, 5
                                     ranks, 3 info tokens, 1 life token and can see all cards.
        num_players (int) -- number of players.
    """
    if environment_name in ["Hanabi-Full", "Hanabi-Full-CardKnowledge"]:
        config = {
            "colors":
                5,
            "ranks":
                5,
            "players":
                num_players,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        }
    elif environment_name == "Hanabi-Full-Minimal":
        config = {
            "colors": 5,
            "ranks": 5,
            "players": num_players,
            "max_information_tokens": 8,
            "max_life_tokens": 3,
            "observation_type": pyhanabi.AgentObservationType.MINIMAL.value
        }
    elif environment_name == "Hanabi-Small":
        config = {
            "colors":
                2,
            "ranks":
                5,
            "players":
                num_players,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        }
    elif environment_name == "Hanabi-Very-Small":
        config = {
            "colors":
                1,
            "ranks":
                5,
            "players":
                num_players,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        }
    elif environment_name == "Hanabi-Very-Small-Oracle":
        config = {
            "colors":
                1,
            "ranks":
                5,
            "players":
                num_players,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.SEER.value
        }
    else:
        raise ValueError("Unknown environment {}".format(environment_name))
    env = rl_env.HanabiEnv(config=config)
    return config, env.num_moves(), env.vectorized_observation_shape()
