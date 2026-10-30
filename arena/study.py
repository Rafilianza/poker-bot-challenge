"""Balanced table-size schedules and results, without changing the poker engine."""

from collections import Counter
from itertools import combinations
import random


def build_schedule(agent_ids, table_sizes, seeds, batches=1):
    """Give each bot equal appearances within every requested table size."""
    if not isinstance(agent_ids, list) or not all(isinstance(key, str) for key in agent_ids):
        raise ValueError('Choose a list of agent IDs')
    if not 4 <= len(agent_ids) <= 80 or len(set(agent_ids)) != len(agent_ids):
        raise ValueError('Choose 4–80 distinct agents for a table-size study')
    if not isinstance(table_sizes, list) or not table_sizes:
        raise ValueError('Select at least one table size')
    if any(type(size) is not int or size not in (4, 5, 6) for size in table_sizes):
        raise ValueError('Table sizes must be 4, 5 or 6')
    if len(set(table_sizes)) != len(table_sizes):
        raise ValueError('Select each table size only once')
    if max(table_sizes) > len(agent_ids):
        raise ValueError('Select enough agents for the largest table size')
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('Provide at least one seed')
    if not all(isinstance(seed, str) and seed for seed in seeds):
        raise ValueError('Seeds must be non-empty strings')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Use distinct seeds')
    if type(batches) is not int or batches < 1:
        raise ValueError('Batches must be a positive integer')

    # Sorting makes a saved experiment independent of checkbox selection order.
    agent_ids = sorted(agent_ids)
    schedule = []
    pair_counts = {size: Counter() for size in table_sizes}
    round_number = 0
    for seed in sorted(seeds):
        for batch in range(1, batches + 1):
            round_number += 1
            for table_size in sorted(table_sizes):
                block_seed = f'{seed}:study-v1:batch:{batch}:size:{table_size}'
                random_source = random.Random(block_seed)
                best_lineups = None
                best_pairs = None
                best_score = None

                # Cyclic windows guarantee equal appearances. Try a few shuffled
                # cycles to reduce repeated opponents across successive batches.
                for candidate in range(12):
                    order = agent_ids.copy()
                    random_source.shuffle(order)
                    lineups = []
                    starts = 1 if len(order) == table_size else len(order)
                    for start in range(starts):
                        lineup = []
                        for position in range(table_size):
                            lineup.append(order[(start + position) % len(order)])
                        lineups.append(lineup)

                    added_pairs = Counter()
                    for lineup in lineups:
                        added_pairs.update(combinations(sorted(lineup), 2))
                    score = 0
                    for pair, count in added_pairs.items():
                        previous = pair_counts[table_size][pair]
                        score += (previous + count) ** 2 - previous ** 2
                    if best_score is None or score < best_score:
                        best_score = score
                        best_lineups = lineups
                        best_pairs = added_pairs

                pair_counts[table_size].update(best_pairs)
                for lineup_number, lineup in enumerate(best_lineups, 1):
                    schedule.append({
                        'round': round_number,
                        'batch': batch,
                        'seed_group': seed,
                        'table': len(schedule) + 1,
                        'table_size': table_size,
                        'seed': f'{block_seed}:lineup:{lineup_number}',
                        'agent_ids': lineup,
                    })
    return schedule


def build_plan(agent_ids, table_sizes, seeds, deals=100, batches=1):
    """Return the complete schedule and its cost before anything is queued."""
    if type(deals) is not int or not 1 <= deals <= 1000:
        raise ValueError('Hands per game must be an integer from 1 to 1000')
    schedule = build_schedule(agent_ids, table_sizes, seeds, batches)
    coverage = {}
    for agent_id in sorted(agent_ids):
        for table_size in sorted(table_sizes):
            coverage[agent_id, table_size] = {
                'id': agent_id, 'table_size': table_size,
                'sets': 0, 'games': 0, 'hands': 0,
            }
    total_games = 0
    for table in schedule:
        table_size = table['table_size']
        total_games += table_size
        for agent_id in table['agent_ids']:
            row = coverage[agent_id, table_size]
            row['sets'] += 1
            row['games'] += table_size
            row['hands'] += table_size * deals
    return {
        'schedule_version': 1,
        'schedule': schedule,
        'total_tables': len(schedule),
        'total_games': total_games,
        'total_hands': total_games * deals,
        'batches': len(seeds) * batches,
        'coverage': list(coverage.values()),
    }


def summarize(run, agents):
    """Recalculate per-size results from saved games, including partial runs."""
    schedule = run.get('schedule', [])
    table_sizes = run['config']['table_sizes']
    big_blind = run.get('profile_config', {}).get('blinds', [1, 2])[1]
    rows = {}
    for agent in agents:
        for table_size in sorted(table_sizes):
            rows[agent['id'], table_size] = {
                'id': agent['id'], 'name': agent['name'], 'table_size': table_size,
                'chips': 0, 'hands': 0, 'games': 0, 'completed_sets': 0,
                'failures': 0, 'decisions': 0, 'bb_per_100': None,
                'completed_batches': 0,
                'batch_min_bb_per_100': None, 'batch_max_bb_per_100': None,
            }

    # A batch contributes to the range only after every duplicate set at this
    # size is complete. Related seat rotations are never separate observations.
    planned_tables = {}
    expected_blocks = {}
    for table in schedule:
        planned_tables[table['table']] = table
        block = (table['seed_group'], table['batch'], table['table_size'])
        expected_blocks.setdefault(block, set()).add(table['table'])

    completed_tables = set()
    block_results = {}
    completed_games = 0
    completed_hands = 0
    for table in run.get('results', []):
        table_number = table['table']
        plan = planned_tables[table_number]
        table_size = plan['table_size']
        block = (plan['seed_group'], plan['batch'], table_size)
        block_rows = block_results.setdefault(block, {})
        games = table.get('games', [])
        is_complete = len(games) == table_size
        if is_complete:
            completed_tables.add(table_number)
        completed_games += len(games)
        completed_hands += sum(game['num_hands'] for game in games)

        for player, agent in enumerate(table['agents']):
            agent_id = agent['id']
            row = rows[agent_id, table_size]
            batch_row = block_rows.setdefault(agent_id, {'chips': 0, 'hands': 0})
            row['completed_sets'] += int(is_complete)
            for game in games:
                chips = game['chips'][player]
                hands = game['num_hands']
                row['chips'] += chips
                row['hands'] += hands
                row['games'] += 1
                row['failures'] += int(game['verdicts'][player] != 'OK')
                teacher_counts = game.get('training', {}).get('teacher_counts', {})
                row['decisions'] += teacher_counts.get(agent_id, 0)
                batch_row['chips'] += chips
                batch_row['hands'] += hands

    batch_rates = {key: [] for key in rows}
    for block, expected_tables in expected_blocks.items():
        if not expected_tables.issubset(completed_tables):
            continue
        for agent_id, batch_row in block_results.get(block, {}).items():
            row_key = (agent_id, block[2])
            rows[row_key]['completed_batches'] += 1
            if batch_row['hands']:
                rate = batch_row['chips'] / big_blind / batch_row['hands'] * 100
                batch_rates[row_key].append(rate)

    for key, row in rows.items():
        if row['hands']:
            row['bb_per_100'] = row['chips'] / big_blind / row['hands'] * 100
        if len(batch_rates[key]) >= 2:
            row['batch_min_bb_per_100'] = min(batch_rates[key])
            row['batch_max_bb_per_100'] = max(batch_rates[key])

    total_games = sum(table['table_size'] for table in schedule)
    return {
        'rows': list(rows.values()),
        'total_tables': len(schedule),
        'total_games': total_games,
        'total_hands': total_games * run['config']['deals'],
        'completed_tables': len(completed_tables),
        'completed_games': completed_games,
        'completed_hands': completed_hands,
    }
