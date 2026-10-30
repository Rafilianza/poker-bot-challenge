"""Convert a supported PHH hand into the arena's existing cloning records.

Only explicit betting actions are replayed. PokerKit's HandHistory iterator is
deliberately not used: it can repair incomplete histories by inventing actions.
Amounts are scaled to a two-chip big blind without rounding; the existing seven
action classes still approximate raise sizes. Outcomes only select whole player
hands and are never included in observations or records.
"""

from copy import deepcopy
import hashlib
from importlib import metadata
import math
from pathlib import Path
import warnings

from arena.bc_features import STREETS, action_target, encode_observation


CONVERTER_VERSION = 1


def _pokerkit():
    """Use the optional PokerKit dependency installed in this environment."""
    try:
        import pokerkit
    except ModuleNotFoundError as error:
        if error.name != 'pokerkit':
            raise
        raise RuntimeError('PHH import needs PokerKit. Run: python -m pip install -r requirements-phh.txt') from error
    # Read metadata beside the actually loaded module, including an earlier
    # import by another caller. A different sys.path entry must not hide a mismatch.
    module_folder = str(Path(pokerkit.__file__).resolve().parent.parent)
    installed_version = None
    for distribution in metadata.distributions(path=[module_folder]):
        if distribution.metadata.get('Name', '').casefold() == 'pokerkit':
            installed_version = distribution.version
            break
    if installed_version != '0.7.6':
        raise RuntimeError(f'PHH replay requires PokerKit 0.7.6; loaded version is {installed_version}')
    return pokerkit


def _finite_numbers(values, count):
    return (
        isinstance(values, list) and len(values) == count
        and all(type(value) in (int, float) and math.isfinite(value) for value in values)
    )


def _split_pluribus_pot(amount, player_count):
    """Pluribus assigns fractional ties equally, including half-chip payouts."""
    return amount / player_count, 0


def _known_holes(actions, player_count, Card):
    """Recover each player's own cards, including cards explicitly shown later."""
    dealt_cards = [[] for _ in range(player_count)]
    shown_cards = [[] for _ in range(player_count)]
    for action in actions:
        words = action.split('#', 1)[0].split()
        if len(words) == 4 and words[:2] == ['d', 'dh']:
            player = words[2]
            cards = list(Card.parse(words[3]))
            destination = dealt_cards
        elif len(words) == 3 and words[1] == 'sm' and words[2] != '-':
            player = words[0]
            cards = list(Card.parse(words[2]))
            destination = shown_cards
        else:
            continue
        if not player.startswith('p') or not player[1:].isdigit():
            raise ValueError('Invalid PHH player label')
        seat = int(player[1:]) - 1
        if not 0 <= seat < player_count:
            raise ValueError('PHH player label is outside the table')
        known = [repr(card) for card in cards if card]
        if len(set(known)) != len(known):
            raise ValueError('A player has duplicate hole cards')
        existing = [repr(card) for card in dealt_cards[seat] + shown_cards[seat] if card]
        if len(set(existing + known)) > 2:
            raise ValueError('Shown cards disagree with the originally dealt cards')
        if destination is dealt_cards:
            destination[seat].extend(cards)
        else:
            for card in cards:
                if card and card not in destination[seat]:
                    destination[seat].append(card)

    known_holes = []
    for dealt, shown in zip(dealt_cards, shown_cards):
        if len(dealt) > 2 or len(shown) > 2:
            raise ValueError('Hold’em players must have two hole cards')
        known_dealt = [repr(card) for card in dealt if card]
        known_shown = [repr(card) for card in shown if card]
        if len(known_shown) == 2 and not set(known_dealt).issubset(known_shown):
            raise ValueError('Shown cards disagree with the originally dealt cards')
        known = known_shown if len(known_shown) == 2 else known_dealt
        if len(set(known)) != len(known):
            raise ValueError('A player has duplicate hole cards')
        known_holes.append(known if len(known) == 2 else [])
    return known_holes


def _observation(state, seat, hole, history, hand_number, scale, big_blind,
                 minimum_increment, blocked_raisers):
    """Read public state plus this actor's private cards into the SDK schema."""
    current_bet = max(state.bets)
    maximum = state.bets[seat] + state.stacks[seat]
    minimum = current_bet + minimum_increment if current_bet else big_blind
    active_stacks = sum(active and stack > 0 for active, stack in zip(state.statuses, state.stacks))
    can_raise = seat not in blocked_raisers and maximum > current_bet and active_stacks > 1
    # Reject rule differences instead of training a mask that the SDK cannot use.
    if can_raise != state.can_complete_bet_or_raise_to():
        raise ValueError('PHH raise reopening rules differ from the arena SDK')
    board = list(state.get_board_cards(0))
    if any(not card for card in board):
        raise ValueError('A decision has unknown public board cards')
    return {
        'type': 'act', 'hand': hand_number, 'seat': seat,
        'street': STREETS[state.street_index],
        'board': [repr(card) for card in board], 'hole': list(hole),
        'pot': state.total_pot_amount * scale,
        'to_call': min(current_bet - state.bets[seat], state.stacks[seat]) * scale,
        'min_raise_to': min(minimum, maximum) * scale,
        'max_raise_to': maximum * scale, 'can_raise': can_raise,
        'stacks': [amount * scale for amount in state.stacks],
        'street_bets': [amount * scale for amount in state.bets],
        'folded': [not active for active in state.statuses],
        'button': len(state.stacks) - 1,
        'history': deepcopy(history), 'players': list(range(len(state.stacks))),
        # PHH files do not provide the live per-decision time bank. It is not a feature.
        'clock_ms': 0,
    }


def convert_hand(hand, *, source_id, group_id, teacher_name='Pluribus',
                 profitable_only=False, source_namespace='phh'):
    """Return native records or a clear reason why this hand cannot be imported.

    ``source_id`` identifies this hand's content, and ``group_id`` identifies its
    source session for grouped train/validation splits. Neither may depend on the
    import job or profit filter. ``source_namespace`` identifies the collection,
    so a named teacher has the same identity across that collection's hands.
    Use ``teacher_name='*'`` to select all named players.
    """
    result = {
        'records': [], 'teachers': [], 'skip_reason': None, 'skip_detail': None,
        'counters': {'hands_seen': 1, 'hands_converted': 0, 'teacher_decisions': 0,
                     'missing_hole_decisions': 0, 'excluded_unprofitable_decisions': 0,
                     'records': 0},
        'raise_errors_bb': [],
    }

    def skip(reason, detail):
        result.update(records=[], skip_reason=reason, skip_detail=detail, raise_errors_bb=[])
        return result

    if not isinstance(hand, dict):
        return skip('invalid_hand', 'A PHH hand must be a TOML table')
    if hand.get('variant') != 'NT':
        return skip('unsupported_variant', 'Only no-limit Texas hold’em (NT) is supported')
    starting_stacks = hand.get('starting_stacks')
    player_count = len(starting_stacks) if isinstance(starting_stacks, list) else 0
    if player_count not in (4, 5, 6):
        return skip('unsupported_table_size', 'Only four, five, or six players are supported')
    if not _finite_numbers(starting_stacks, player_count) or any(value <= 0 for value in starting_stacks):
        return skip('unsupported_stacks', 'Starting stacks must be known, finite, and positive')
    antes = hand.get('antes')
    if not _finite_numbers(antes, player_count) or any(antes):
        return skip('unsupported_antes', 'Only explicitly recorded zero antes are supported')
    blinds = hand.get('blinds_or_straddles')
    if (not _finite_numbers(blinds, player_count) or blinds[0] <= 0
            or blinds[1] != 2 * blinds[0] or any(blinds[2:])):
        return skip('unsupported_blinds', 'Require standard small/big blinds, with no straddles')
    big_blind = blinds[1]
    if hand.get('min_bet') != big_blind:
        return skip('unsupported_min_bet', 'The minimum bet must equal the big blind')
    if any(not math.isclose(value, 100 * big_blind, rel_tol=1e-12) for value in starting_stacks):
        return skip('unsupported_stack_depth', 'Every player must start with exactly 100 big blinds')
    if not all(isinstance(value, str) and value for value in (source_id, group_id, source_namespace)):
        raise ValueError('Stable source_id, group_id, and source_namespace strings are required')
    players = hand.get('players')
    if (not isinstance(players, list) or len(players) != player_count
            or not all(isinstance(name, str) for name in players)):
        return skip('missing_players', 'Player names are required to select stable teachers')
    if not isinstance(teacher_name, str) or not teacher_name.strip():
        raise ValueError('Choose a teacher name, or * for all named players')
    selected_seats = [seat for seat, name in enumerate(players)
                      if name.strip() and (teacher_name == '*' or name.casefold() == teacher_name.casefold())]
    if not selected_seats:
        return skip('teacher_not_present', 'The selected teacher is not in this hand')
    actions = hand.get('actions')
    if not isinstance(actions, list) or not actions or not all(isinstance(action, str) for action in actions):
        return skip('invalid_actions', 'A nonempty list of explicit PHH actions is required')

    # Profit is checked only after replay. In profit mode a supplied outcome is
    # also required, because an incomplete history can contain finishing_stacks.
    finishing_stacks = hand.get('finishing_stacks')
    if profitable_only and finishing_stacks is None:
        return skip('missing_outcome', 'Profitable-only import requires finishing_stacks')
    for field in ('finishing_stacks', 'winnings'):
        values = hand.get(field)
        if values is not None and (not _finite_numbers(values, player_count) or any(value < 0 for value in values)):
            return skip('invalid_outcome', f'{field} must contain finite non-negative values for every player')

    pokerkit = _pokerkit()
    from pokerkit.notation import parse_action

    scale = 2 / big_blind
    hand_number = hand.get('hand', 0)
    teachers = {}
    for seat in selected_seats:
        identity = f'phh:{source_namespace}:{players[seat].casefold()}'
        identifier = hashlib.sha256(identity.encode()).hexdigest()
        version_hash = hashlib.sha256(f'{identity}:converter:{CONVERTER_VERSION}'.encode()).hexdigest()
        teachers[seat] = {'id': 'phh-' + identifier[:24], 'name': players[seat], 'hash': version_hash}
    result['teachers'] = list({teacher['id']: teacher for teacher in teachers.values()}.values())

    try:
        known_holes = _known_holes(actions, player_count, pokerkit.Card)
        history_object = pokerkit.HandHistory(
            variant='NT', ante_trimming_status=hand.get('ante_trimming_status', False),
            antes=antes, blinds_or_straddles=blinds, min_bet=big_blind,
            starting_stacks=starting_stacks, actions=actions,
        )
        if source_namespace == 'pluribus':
            # The source uses exact equal shares, not a positional odd chip.
            history_object.divmod = _split_pluribus_pot
        state = history_object.create_state()
        public_history = []
        pending_records = []
        previous_street = None
        minimum_increment = big_blind
        blocked_raisers = set()
        acted_seats = set()

        # Warnings about invalid cards/folding are errors, not silent repairs.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for action_index, action in enumerate(actions):
                words = action.split('#', 1)[0].split()
                if not words:
                    continue
                is_decision = len(words) >= 2 and words[1] in ('f', 'cc', 'cbr')
                if state.can_burn_card():
                    state.burn_card('??')
                if is_decision:
                    # Missing private deals mean unknown cards, never random cards.
                    while state.can_deal_hole():
                        state.deal_hole('??')
                    seat = state.actor_index
                    if seat is None or words[0] != f'p{seat + 1}':
                        raise ValueError(f'Action {action_index + 1} is out of turn')
                    if state.street_index != previous_street:
                        minimum_increment = big_blind
                        blocked_raisers.clear()
                        acted_seats.clear()
                        previous_street = state.street_index
                    observation = _observation(
                        state, seat, known_holes[seat], public_history, hand_number,
                        scale, big_blind, minimum_increment, blocked_raisers,
                    )
                    if words[1] == 'cbr':
                        amount = pokerkit.parse_value(words[2])
                        if not math.isfinite(amount):
                            raise ValueError('Raise amount must be finite')
                        executed = {'action': 'raise', 'amount': amount * scale}
                    elif words[1] == 'f':
                        executed = {'action': 'fold', 'amount': 0}
                    else:
                        kind = 'call' if observation['to_call'] else 'check'
                        executed = {'action': kind, 'amount': observation['to_call']}
                    # Validation uses the unchanged seven-class label mapper.
                    target, raise_error = action_target(observation, executed)
                    old_current_bet = max(state.bets)

                parse_action(state, action)

                if not is_decision:
                    continue
                if seat in teachers:
                    result['counters']['teacher_decisions'] += 1
                    if not known_holes[seat]:
                        result['counters']['missing_hole_decisions'] += 1
                    else:
                        encode_observation(observation)
                        record = {
                            'schema_version': 1, 'run_id': 'phh:' + source_namespace,
                            'game_id': source_id, 'seed_group': group_id,
                            'game_seed': source_id, 'offset': 0,
                            'decision_id': f'phh:{source_id}:{len(public_history)}',
                            'teacher': {**teachers[seat], 'player': seat},
                            'observation': observation, 'requested_action': dict(executed),
                            'executed_action': executed, 'verdict': 'OK',
                            'table_size': player_count,
                            'source': {'kind': 'phh', 'namespace': source_namespace,
                                       'hand_digest': source_id, 'converter_version': CONVERTER_VERSION},
                        }
                        pending_records.append((seat, record, raise_error / 2))
                public_history.append([observation['street'], seat, executed['action'], executed['amount']])
                acted_seats.add(seat)
                if executed['action'] == 'raise':
                    increment = amount - old_current_bet
                    if increment >= minimum_increment:
                        minimum_increment = increment
                        blocked_raisers.clear()
                        acted_seats = {seat}
                    else:
                        blocked_raisers.update(acted_seats)

            # Known hands can be evaluated if explicit show actions were omitted.
            # An unknown hand is never automatically mucked to manufacture a winner.
            while state.showdown_index is not None:
                seat = state.showdown_index
                if len(state.hole_cards[seat]) != 2 or not all(state.hole_cards[seat]):
                    return skip('unresolved_showdown', 'An unshown live hand prevents reliable outcome verification')
                state.show_or_muck_hole_cards(True)

        if state.status:
            return skip('incomplete_hand', 'Explicit actions do not reach a completed, resolved hand')
        if finishing_stacks is not None:
            for expected, actual in zip(finishing_stacks, state.stacks):
                if not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=big_blind * 1e-9):
                    return skip('inconsistent_outcome', 'finishing_stacks disagree with the complete replay')
        if hand.get('winnings') is not None:
            awards = [0] * player_count
            for operation in state.operations:
                if isinstance(operation, pokerkit.ChipsPushing):
                    for seat, amount in enumerate(operation.amounts):
                        awards[seat] += amount
            if any(not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=big_blind * 1e-9)
                   for expected, actual in zip(hand['winnings'], awards)):
                return skip('inconsistent_outcome', 'winnings disagree with replayed collected pots')

        for seat, record, raise_error_bb in pending_records:
            if profitable_only and finishing_stacks[seat] <= starting_stacks[seat]:
                result['counters']['excluded_unprofitable_decisions'] += 1
                continue
            result['records'].append(record)
            if record['executed_action']['action'] == 'raise':
                result['raise_errors_bb'].append(raise_error_bb)
        if not result['records']:
            if result['counters']['excluded_unprofitable_decisions']:
                return skip('no_profitable_teacher_decisions', 'Selected players with known cards did not finish with positive net profit')
            if result['counters']['missing_hole_decisions']:
                return skip('missing_teacher_cards', 'Selected teacher decisions have no recoverable private cards')
            return skip('no_teacher_decisions', 'The selected teacher has no explicit betting decisions')
        result['counters']['hands_converted'] = 1
        result['counters']['records'] = len(result['records'])
        return result
    except (ValueError, TypeError, IndexError, KeyError, AssertionError, UserWarning) as error:
        return skip('invalid_replay', str(error))
