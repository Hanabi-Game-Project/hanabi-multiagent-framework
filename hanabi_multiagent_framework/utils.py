"""
A collection of helper functions
"""
from hanabi_learning_environment import pyhanabi, rl_env
import numpy as np
from scipy import stats

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
    elif environment_name == "Hanabi-Full-Oracle":
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
    elif environment_name in ["Hanabi-Small", "Hanabi-Small-CardKnowledge"]:
        config = {
            "colors":
                3,
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
    elif environment_name == "Hanabi-Small-Oracle":
        config = {
            "colors":
                3,
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
    return {k : str(v) for k, v in config.items()}



class ObservationCanonicalDecoder:
    """Helper class to convert one-hot encoded observations into human-readable form.
    """
    def __init__(self, game):
        self._game = game
        self.num_players = self._game.num_players()
        self.num_colors = self._game.num_colors()
        self.num_ranks = self._game.num_ranks()
        self.hand_size = self._game.hand_size()
        self.max_deck_size = self.num_colors * self._game.cards_per_color()
        self.max_info_tokens = self._game.max_information_tokens()
        self.max_life_tokens = self._game.max_life_tokens()

    def decode(self, obs):
        """Decode a single one-hot encoded observation.

        Args:
            obs -- encoded observation.
        Retuns an observation as a dictionary with following keys:
            hands .... tbc
        """
        offset = 0
        offset, hands = self._decode_hands(obs, offset)
        offset, board = self._decode_board(obs, offset)
        offset, discards = self._decode_discards(obs, offset)
        offset, last_action = self._decode_last_action(obs, offset)
        if self._game.observation_type() != pyhanabi.AgentObservationType.Minimal:
            offset, card_knowledge = self._decode_card_knowledge(obs, offset)
        else:
            card_knowledge = None

        assert offset == len(obs)
        return {"hands" : hands, "board" : board, "discards" : discards,
                "last_action" : last_action, "card_knowledge" : card_knowledge}

    def _decode_hands(self, obs, offset):
        hands = []
        bits_per_card = self.num_ranks * self.num_colors
        for player_id in range(self.num_players):
            hands.append([])
            for card_idx in range(self.hand_size):
                card = {}
                for card_bit_idx in range(bits_per_card):
                    if obs[offset + card_bit_idx] == 1:
                        card['rank'] = card_bit_idx % self.num_ranks
                        card['color'] = (card_bit_idx - card['rank']) // self.num_ranks
                        break
                hands[player_id].append(card)
                offset += bits_per_card

        assert offset == bits_per_card * self.num_players * self.hand_size

        for player_id in range(self.num_players):
            assert len(hands[player_id]) + obs[offset + player_id] == self.hand_size

        offset += self.num_players

        return offset, hands

    def _decode_board(self, obs, offset):
        init_offset = offset
        cards_on_hands = self.num_players * self.hand_size
        deck_size = obs[offset : offset + self.max_deck_size - cards_on_hands].sum()
        offset += self.max_deck_size - cards_on_hands
        fireworks = [0 for _ in range(self.num_colors)]
        for color_id in range(self.num_colors):
            for rank_id in range(self.num_ranks):
                if obs[offset + rank_id] == 1:
                    fireworks[color_id] = rank_id + 1
            offset += self.num_ranks

        info_tokens = obs[offset : offset + self.max_info_tokens].sum()
        offset += self.max_info_tokens

        life_tokens = obs[offset : offset + self.max_life_tokens].sum()
        offset += self.max_life_tokens

        assert offset - init_offset == (self.max_deck_size - cards_on_hands +
                                        self.num_colors * self.num_ranks +
                                        self.max_info_tokens +
                                        self.max_life_tokens)
        return offset, {"deck_size" : deck_size, "fireworks" : fireworks,
                        "info_tokens" : info_tokens, "life_tokens" : life_tokens}

    def _decode_discards(self, obs, offset):
        init_offset = offset

        discard_pile = []
        for color in self.num_colors:
            for rank in self.num_ranks:
                num_cards = self._game.num_cards(color, rank)
                num_discarded = obs[offset : offset + num_cards].sum()
                discard_pile.extend([{"rank": rank, "color": color} for _ in range(num_discarded)])
                offset += num_cards
        assert offset - init_offset == self.max_deck_size
        return offset, discard_pile

    def _decode_last_action(self, obs, offset):
        move_types = ["PLAY", "DISCARD", "REVEAL_COLOR", "REVEAL_RANK"]
        init_offset = offset
        last_action_length = (self.num_players + # current player
                              len(move_types)  + # move type (play, dis, rev col, rev rank)
                              self.num_players + # target player (for hints)
                              self.num_colors  + # color (for hints)
                              self.num_ranks   + # rank (for hints)
                              self.hand_size   + # outcome (for hints)
                              self.hand_size   + # position (for play)
                              self.num_ranks * self.num_colors + # card (for play or discard)
                              2)                 # play (successful, added info token)
        if obs[offset : offset + last_action_length].sum() == 0:
            return offset + last_action_length, {}

        last_action = {}
        for player_id in range(self.num_players):
            if obs[offset + player_id] == 1:
                last_action['cur_player'] = player_id
                break
        offset += self.num_players

        for move_id, move_type in enumerate(move_types):
            if obs[offset + move_id] == 1:
                last_action['move'] = move_type
                if move_type in ["REVEAL_COLOR", "REVEAL_RANK"]:
                    is_hint = True
                break
        offset += len(move_types)

        if is_hint:
            for player_id in range(self.num_players):
                if obs[offset + player_id] == 1:
                    last_action['trg_player'] = player_id
                    break
        offset += self.num_players

        if is_hint:
            for color in range(self.num_colors):
                if obs[offset + color] == 1:
                    last_action['hint_color'] = color
                    break
        offset += self.num_colors

        if is_hint:
            for rank in range(self.num_ranks):
                if obs[offset + rank] == 1:
                    last_action['hint_rank'] = rank
                    break
        offset += self.num_colors

        if is_hint:
            for hinted in range(self.hand_size):
                if obs[offset + hinted] == 1:
                    last_action['outcome'] = hinted
        offset += self.hand_size

        for pos in range(self.hand_size):
            if obs[offset + pos] == 1:
                last_action["position"] = pos
                break
        offset += self.hand_size

        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                if obs[offset + rank] == 1:
                    last_action["card"] = {"color" : color, "rank" : rank}

        last_action["success"] = obs[offset] == 1
        offset += 1
        last_action["info_tokens_added"] = obs[offset] == 1
        offset += 1

        assert offset - init_offset == last_action_length
        return offset, last_action

    def _decode_card_knowledge(self, obs, offset):
        card_knowledge = {}
        return offset, card_knowledge

def print_hist(reward, max_bin, height=10):
    h, b = np.histogram(reward, range(0, int(max_bin) + 1))
    scale = height / len(reward)
    cols = []
    for h_ in h:
        counts = int(h_ * scale)
        cols.append(list(('#' * counts) + (' ' * (height - counts))))
    cols = np.array(cols)
    for row_idx in range(height // 2, -1, -1):
        print(f"{int((row_idx + 1) / scale):6}" + "| " + "  ".join(cols[:, row_idx]))
    print("      -" + "---" * int(max_bin + 1))
    print("       " + " ".join([f"{int(b_):2}" for b_ in b]))


def eval_pretty_print(step_rewards, total_reward):
    steps_descr      = "| Step       |"
    terminated_descr = "| Terminated |"
    means_descr      = "| Mean Rew   |"
    stds_descr       = "| Std Rew    |"
    mins_descr       = "| Min Rew    |"
    maxs_descr       = "| Max Rew    |"

    n_steps = len(step_rewards)
    n_rows = n_steps // 12
    if n_steps - n_rows * 12 > 0:
        n_rows += 1

    steps = ["" for _ in range(n_rows)]
    terminated = ["" for _ in range(n_rows)]
    means = ["" for _ in range(n_rows)]
    stds = ["" for _ in range(n_rows)]
    mins = ["" for _ in range(n_rows)]
    maxs = ["" for _ in range(n_rows)]

    for i, step in enumerate(step_rewards):
        row = i // 12
        steps[row]      += f" {i+1:6} |"
        terminated[row] += f" {step['terminated']:6} |"
        means[row]      += f" {step['rewards'].mean():6.3f} |"
        stds[row]       += f" {step['rewards'].std():6.3f} |"
        mins[row]       += f" {int(step['rewards'].min()):6} |"
        maxs[row]       += f" {int(step['rewards'].max()):6} |"
    border = '-' * (len(steps[0]) + len(steps_descr))
    for st, te, me, sd, mi, ma in zip(steps, terminated, means, stds, mins, maxs):
        print(border)
        print(steps_descr + st)
        print(terminated_descr + te)
        print(means_descr + me)
        print(stds_descr + sd)
        print(mins_descr + mi)
        print(maxs_descr + ma)
        print(border)
    print(f"Total: mean {total_reward.mean():.3f} med {np.median(total_reward):.0f} mode(s) {stats.mode(total_reward, axis=None)[0]} std {total_reward.std():.3f} min {int(total_reward.min()):2} max {int(total_reward.max()):2}")
    print_hist(total_reward, 25, height=20)
    print('')
