"""Public game context shared by placement PPO and its exported SDK bot.

The original 189 BC features stay unchanged. Seven extra features describe
completed hands / total hands, followed by cumulative chip scores in relative
seat order. Scores use chips / (starting stack * total hands), so their units
are consistent throughout a game. With six players, a score can reach five;
scores are not clipped. Only completed public hand results update them.
"""
import math

if __package__:
    from .bc_features import FEATURE_COUNT, encode_observation
else:
    from bc_features import FEATURE_COUNT, encode_observation


CONTEXT_VERSION = 1
CONTEXT_FEATURE_COUNT = 7
PLACEMENT_FEATURE_COUNT = FEATURE_COUNT + CONTEXT_FEATURE_COUNT


class GameContext:
    """Remember public scores within one game, resetting at every SDK hello."""

    def __init__(self):
        self.player_id = None
        self.total_hands = 0
        self.initial_stack = 0
        self.scores = []
        self.last_completed_hand = -1

    def send(self, message):
        message_type = message['type']
        if message_type == 'hello':
            player_count = message['num_players']
            total_hands = message['num_hands']
            initial_stack = message['stack']
            player_id = message['player']
            if type(player_count) is not int or not 2 <= player_count <= 6:
                raise ValueError('Game context supports two through six players')
            if type(total_hands) is not int or total_hands < 1:
                raise ValueError('Game context needs a positive hand count')
            if type(initial_stack) is not int or initial_stack < 1:
                raise ValueError('Game context needs a positive starting stack')
            if type(player_id) is not int or not 0 <= player_id < player_count:
                raise ValueError('Game context received an invalid player ID')
            self.player_id = player_id
            self.total_hands = total_hands
            self.initial_stack = initial_stack
            self.scores = [0.0] * player_count
            self.last_completed_hand = -1
        elif message_type == 'hand_end':
            if not self.scores:
                raise ValueError('Game context needs hello before a hand result')
            players = message['players']
            deltas = message['deltas']
            if sorted(players) != list(range(len(self.scores))) or len(deltas) != len(players):
                raise ValueError('Hand results must map each seat to one player')
            if message['hand'] <= self.last_completed_hand:
                raise ValueError('A completed hand cannot update game scores twice')
            if not all(type(delta) in (int, float) and math.isfinite(delta) for delta in deltas):
                raise ValueError('Hand results contain invalid chip changes')
            # Seat positions rotate; stable player IDs keep each score attached
            # to the same opponent throughout the game.
            for seat, player_id in enumerate(players):
                self.scores[player_id] += deltas[seat]
            self.last_completed_hand = message['hand']

    def encode(self, observation):
        if not self.scores:
            raise ValueError('Game context needs hello before an action')
        players = observation['players']
        seat = observation['seat']
        if len(players) != len(self.scores) or players[seat] != self.player_id:
            raise ValueError('The observation belongs to a different game or player')
        features = encode_observation(observation)
        features.append(observation['hand'] / self.total_hands)
        score_scale = self.initial_stack * self.total_hands
        for relative_seat in range(6):
            if relative_seat < len(players):
                player_seat = (seat + relative_seat) % len(players)
                player_id = players[player_seat]
                features.append(self.scores[player_id] / score_scale)
            else:
                features.append(0.0)
        return features
