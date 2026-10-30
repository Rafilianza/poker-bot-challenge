"""Shared game rewards and exact duplicate-round selection scores."""
import math

from arena.scoring import points

SELECTION_METRIC = 'placement_chip_score'


def reward_settings(config):
    mode = config.get('reward_mode', 'chips')
    if mode not in ('chips', 'game_placement'):
        raise ValueError('Choose chips or game_placement reward mode')
    weight = config.get('placement_weight', 0.9)
    scale = config.get('chip_scale_bb100', 100.0)
    if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Placement weight must be between zero and one')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Chip scale must be a positive finite bb/100 value')
    return {'reward_mode': mode, 'placement_weight': float(weight), 'chip_scale_bb100': float(scale)}


def game_reward(chips, learner=0, hands=100, bb=2, placement_weight=0.9, chip_scale_bb100=100):
    """Bounded terminal game reward, with tied ranks sharing their points."""
    settings = reward_settings({'reward_mode': 'game_placement',
                                'placement_weight': placement_weight, 'chip_scale_bb100': chip_scale_bb100})
    if (not isinstance(chips, (list, tuple)) or not 2 <= len(chips) <= 6
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in chips)
            or not math.isclose(sum(chips), 0, abs_tol=1e-6)):
        raise ValueError('Game chips must be a finite zero-sum vector for 2–6 players')
    if type(learner) is not int or not 0 <= learner < len(chips):
        raise ValueError('Learner must be a valid player index')
    if type(hands) is not int or hands < 1:
        raise ValueError('A game reward needs a positive number of completed hands')
    if type(bb) not in (int, float) or not math.isfinite(bb) or bb <= 0:
        raise ValueError('The big blind must be positive and finite')
    game_points = points(chips)
    normalized = (game_points[learner] - 1) / (len(chips) - 1)
    placement = 2 * normalized - 1
    rate = chips[learner] / bb / hands * 100
    chip_component = math.tanh(rate / settings['chip_scale_bb100'])
    reward = settings['placement_weight'] * placement + (1 - settings['placement_weight']) * chip_component
    return {'reward': reward, 'placement': placement, 'chip_component': chip_component,
            'points': game_points, 'bb100': rate, 'normalized_placement': normalized}


def _round_summary(rounds, expected_count):
    complete = [row for row in rounds if row['completed']]
    eligible = bool(expected_count) and len(complete) == expected_count and all(row['eligible_for_best'] for row in complete)
    hands = sum(row['hands'] for row in complete)
    chips = sum(row['chips'] for row in complete)
    result = {'expected_rounds': expected_count, 'completed_rounds': len(complete),
              'eligible_for_best': eligible, 'hands': hands, 'chips': chips,
              'failures': sum(row['failures'] for row in rounds),
              'bb100': chips / 2 / hands * 100 if hands else None,
              'selection_score': None, 'mean_round_points': None,
              'normalized_round_placement': None, 'centered_round_placement': None,
              'chip_component': None}
    if complete:
        for key in ('normalized_round_placement', 'centered_round_placement', 'chip_component'):
            result[key] = sum(row[key] for row in complete) / len(complete)
        result['mean_round_points'] = sum(row['round_points'][0] for row in complete) / len(complete)
    if eligible:
        result['selection_score'] = sum(row['selection_score'] for row in complete) / len(complete)
    return result


def aggregate_rounds(games, placement_weight=0.9, chip_scale_bb100=100,
                     expected_seeds=None, table_sizes=None):
    """Rank summed game points, then average the bounded objective by table size.

    A duplicate round uses the same ordered players and deck seed at every seat
    offset. Incomplete rounds and SDK failures cannot select a best checkpoint.
    Supply expected_seeds and table_sizes when an entire round may be absent.
    """
    settings = reward_settings({'reward_mode': 'game_placement',
                                'placement_weight': placement_weight, 'chip_scale_bb100': chip_scale_bb100})
    grouped = {}
    for game in games:
        size = game.get('table_size')
        offset = game.get('offset')
        seed = game.get('seed_group')
        if type(size) is not int or size not in (2, 3, 4, 5, 6):
            raise ValueError('Every duplicate game needs a valid table size')
        if type(offset) is not int or not 0 <= offset < size:
            raise ValueError('Every duplicate game needs a valid seat offset')
        if type(seed) not in (str, int):
            raise ValueError('Every duplicate game needs its seed group')
        grouped.setdefault((seed, size), []).append(game)
    seeds = sorted({key[0] for key in grouped}, key=str)
    if expected_seeds is not None:
        seeds = list(range(expected_seeds)) if type(expected_seeds) is int else list(expected_seeds)
        if len(seeds) != len(set(seeds)):
            raise ValueError('Expected seed groups must be distinct')
    sizes = sorted({key[1] for key in grouped}) if table_sizes is None else list(table_sizes)
    if len(sizes) != len(set(sizes)) or any(type(size) is not int or size not in (2, 3, 4, 5, 6) for size in sizes):
        raise ValueError('Expected table sizes must be distinct sizes from 2 through 6')
    if any(seed not in seeds or size not in sizes for seed, size in grouped):
        raise ValueError('A game does not belong to the expected duplicate schedule')

    rounds = []
    for seed in seeds:
        for size in sizes:
            selected = grouped.get((seed, size), [])
            offsets = [game['offset'] for game in selected]
            if len(offsets) != len(set(offsets)):
                raise ValueError('A duplicate round repeats a seat offset')
            lineups = []
            for game in selected:
                opponents = game.get('opponents', [])
                lineup = [(opponent.get('id'), opponent.get('hash')) if isinstance(opponent, dict)
                          else (opponent, None) for opponent in opponents]
                lineups.append(lineup)
            if lineups and any(lineup != lineups[0] for lineup in lineups):
                raise ValueError('Duplicate games must keep the same ordered opponent lineup')
            if selected and any(game.get('seed') != selected[0].get('seed') for game in selected):
                raise ValueError('Duplicate games must use the same deck seed')
            failures = sum(sum(verdict != 'OK' for verdict in game.get('verdicts', [])) for game in selected)
            complete = set(offsets) == set(range(size))
            complete = complete and all(game.get('completed') is True for game in selected)
            round_row = {'seed_group': seed, 'table_size': size, 'offsets': sorted(offsets),
                         'completed': complete, 'eligible_for_best': False, 'failures': failures,
                         'hands': 0, 'chips': 0, 'game_points': None, 'round_points': None,
                         'normalized_round_placement': None, 'centered_round_placement': None,
                         'chip_component': None, 'selection_score': None, 'bb100': None}
            if complete:
                if any(len(lineup) != size - 1 or any(identifier is None for identifier, _ in lineup)
                       for lineup in lineups):
                    raise ValueError('A completed duplicate round needs every ordered opponent identifier')
                if any(len(game.get('chips') or []) != size or len(game.get('verdicts') or []) != size for game in selected):
                    raise ValueError('A completed game must include chips and verdicts for every player')
                hand_counts = [game.get('completed_hands') for game in selected]
                if len(set(hand_counts)) != 1 or any(type(hands) is not int or hands <= 0 for hands in hand_counts):
                    raise ValueError('Duplicate games must complete the same positive number of hands')
                if any(game.get('requested_hands', game['completed_hands']) != game['completed_hands'] for game in selected):
                    raise ValueError('A completed duplicate game is missing requested hands')
                game_metrics = [game_reward(game['chips'], hands=game['completed_hands'],
                                           placement_weight=settings['placement_weight'],
                                           chip_scale_bb100=settings['chip_scale_bb100']) for game in selected]
                totals = [sum(metrics['points'][player] for metrics in game_metrics) for player in range(size)]
                round_points = points(totals)
                normalized = (round_points[0] - 1) / (size - 1)
                placement = 2 * normalized - 1
                chip_component = sum(metrics['chip_component'] for metrics in game_metrics) / size
                score = settings['placement_weight'] * placement + (1 - settings['placement_weight']) * chip_component
                hands = sum(hand_counts)
                chips = sum(game['chips'][0] for game in selected)
                round_row.update(hands=hands, chips=chips, game_points=totals, round_points=round_points,
                                 normalized_round_placement=normalized, centered_round_placement=placement,
                                 chip_component=chip_component, selection_score=score, bb100=chips / 2 / hands * 100,
                                 eligible_for_best=failures == 0 and not any(game.get('error') for game in selected))
            rounds.append(round_row)

    by_size = {}
    for size in sizes:
        by_size[str(size)] = _round_summary([row for row in rounds if row['table_size'] == size], len(seeds))
    by_seed = []
    for seed in seeds:
        selected = [row for row in rounds if row['seed_group'] == seed]
        seed_summary = _round_summary(selected, len(sizes))
        if selected and all(row['completed'] for row in selected):
            seed_summary['bb100'] = sum(row['bb100'] for row in selected) / len(selected)
        seed_summary.update(seed_group=seed, by_table_size={str(row['table_size']): row['bb100'] for row in selected})
        by_seed.append(seed_summary)
    result = _round_summary(rounds, len(seeds) * len(sizes))
    for metric in ('bb100', 'normalized_round_placement', 'centered_round_placement', 'chip_component', 'selection_score'):
        values = [row[metric] for row in by_size.values()]
        result[metric] = sum(values) / len(values) if values and all(value is not None for value in values) else None
    result.update(selection_metric=SELECTION_METRIC, reward_config=settings,
                  by_table_size=by_size, by_seed_group=by_seed, rounds=rounds,
                  seed_groups=seeds, aggregation='Equal mean of table-size scores; each round ranks summed per-game points.')
    return result
