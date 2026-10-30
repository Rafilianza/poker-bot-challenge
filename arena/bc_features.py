"""Observable poker features and legal actions shared by training and students.

This module deliberately uses only Python's standard library so it can be copied
into a scaffold submission together with the exported model weights.
"""
import math
import random

FEATURE_VERSION = 1
ACTION_NAMES = ['fold', 'check/call', 'minimum raise', 'half-pot raise',
                'pot raise', 'double-pot raise', 'all-in']
STREETS = ['preflop', 'flop', 'turn', 'river']
FEATURE_COUNT = 189


def encode_observation(observation):
    """Encode only the actor's cards and information visible at decision time."""
    features = []
    for field in ('hole', 'board'):
        cards = [0.0] * 52
        for card in observation[field]:
            rank = '23456789TJQKA'.index(card[0])
            suit = 'cdhs'.index(card[1])
            cards[rank * 4 + suit] = 1.0
        features.extend(cards)
    features.extend(float(observation['street'] == street) for street in STREETS)
    players = len(observation['stacks'])
    seat = observation['seat']
    if players not in (2, 3, 4, 5, 6):
        raise ValueError('The cloning encoder supports 2 through 6 players')
    features.append(players / 6.0)
    button_position = (observation['button'] - seat) % players
    features.extend(float(position == button_position) for position in range(6))
    features.extend([
        observation['pot'] / 1200.0,
        observation['to_call'] / 200.0,
        observation['min_raise_to'] / 200.0,
        observation['max_raise_to'] / 200.0,
        float(observation['can_raise']),
        sum(not folded for folded in observation['folded']) / 6.0,
        len(observation['history']) / 100.0,
        observation['stacks'][seat] / 200.0,
    ])
    # Relative seats let the same features work after the seats rotate.
    for relative_seat in range(6):
        if relative_seat >= players:
            features.extend([0.0] * 11)
            continue
        player_seat = (seat + relative_seat) % players
        actions = [event for event in observation['history'] if event[1] == player_seat]
        calls = sum(event[2] == 'call' for event in actions)
        raises = sum(event[2] == 'raise' for event in actions)
        street_raises = sum(event[2] == 'raise' and event[0] == observation['street']
                            for event in actions)
        last_action = actions[-1][2] if actions else None
        features.extend([
            1.0, observation['stacks'][player_seat] / 200.0,
            observation['street_bets'][player_seat] / 200.0,
            float(observation['folded'][player_seat]),
            calls / 10.0, raises / 10.0, street_raises / 10.0,
        ])
        features.extend(float(last_action == action) for action in ('fold', 'check', 'call', 'raise'))
    if len(features) != FEATURE_COUNT:
        raise ValueError('Unexpected feature length')
    return features


def action_options(observation):
    """Return seven action choices and a mask excluding illegal or duplicate ones."""
    to_call = observation['to_call']
    maximum = observation['max_raise_to']
    minimum = min(observation['min_raise_to'], maximum)
    current = observation['street_bets'][observation['seat']] + to_call
    pot_after_call = observation['pot'] + to_call
    amounts = [minimum]
    for fraction in (0.5, 1.0, 2.0):
        amount = current + round(pot_after_call * fraction)
        amounts.append(max(minimum, min(maximum, amount)))
    amounts.append(maximum)
    options = [{'action': 'fold', 'amount': 0},
               {'action': 'call' if to_call else 'check', 'amount': 0}]
    options.extend({'action': 'raise', 'amount': amount} for amount in amounts)
    mask = [to_call > 0, True] + [False] * 5
    if observation['can_raise'] and maximum > current:
        # Reserve the all-in amount for its own class. Other repeated amounts
        # use the first matching raise class, avoiding ambiguous targets.
        used_amounts = {maximum}
        mask[6] = True
        for action_index in range(2, 6):
            amount = options[action_index]['amount']
            if amount not in used_amounts:
                mask[action_index] = True
                used_amounts.add(amount)
    return options, mask


def action_target(observation, executed_action):
    """Map the action the engine executed to a legal class and raise-size error."""
    options, mask = action_options(observation)
    action = executed_action['action']
    if action == 'fold' and mask[0]:
        return 0, 0.0
    if action in ('check', 'call'):
        if (action == 'check') != (observation['to_call'] == 0):
            raise ValueError('An executed check/call disagrees with the amount to call')
        return 1, 0.0
    if action != 'raise':
        raise ValueError('A demonstration contains an invalid executed action')
    candidates = [index for index in range(2, 7) if mask[index]]
    if not candidates:
        raise ValueError('A demonstration raises when raising is unavailable')
    amount = executed_action['amount']
    if not min(observation['min_raise_to'], observation['max_raise_to']) <= amount <= observation['max_raise_to']:
        raise ValueError('An executed raise is outside the legal bounds')
    target = min(candidates, key=lambda index: abs(options[index]['amount'] - amount))
    return target, float(abs(options[target]['amount'] - amount))


def forward_logits(features, layers):
    """Run a small exported neural network without NumPy or PyTorch."""
    values = features
    for layer_index, layer in enumerate(layers):
        output = []
        for weights, bias in zip(layer['weight'], layer['bias']):
            value = bias + sum(weight * item for weight, item in zip(weights, values))
            if layer_index < len(layers) - 1:
                value = max(0.0, value)
            output.append(value)
        values = output
    return values


def choose_action(observation, logits, rng=None):
    """Sample the learned probabilities after removing illegal actions."""
    options, mask = action_options(observation)
    legal_indices = [index for index, allowed in enumerate(mask) if allowed]
    maximum = max(logits[index] for index in legal_indices)
    probabilities = [math.exp(logits[index] - maximum) for index in legal_indices]
    generator = rng or random
    chosen = generator.choices(legal_indices, weights=probabilities, k=1)[0]
    return options[chosen]
