"""Permanent PPO snapshots and comparable, resumable full-game benchmarks.

The benchmark is deliberately separate from PPO's short validation objective.
Every score comes from the official SDK result, including its failure verdicts.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time

from arena.bc_features import ACTION_NAMES, FEATURE_COUNT, FEATURE_VERSION
from arena.bc_features import choose_action, encode_observation, forward_logits
from arena.scoring import points
from arena.ppo_objectives import aggregate_rounds, reward_settings

DEFAULTS = {'enabled': True, 'interval': 50000, 'eval_hands': 100, 'eval_seeds': 3,
            'table_sizes': [4, 5, 6], 'seed': 42}
SNAPSHOT_ID = re.compile(r'(?:bc|episode|final)-[0-9]{9}')


def request_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def opponent_hash(path):
    """Hash the executable snapshot, excluding Python's disposable bytecode."""
    path = Path(path)
    folder = path.parent if path.is_file() else path
    digest = hashlib.sha256()
    for item in sorted(folder.rglob('*')):
        if item.is_file() and '__pycache__' not in item.parts and item.suffix != '.pyc':
            digest.update(str(item.relative_to(folder)).encode())
            digest.update(b'\0')
            digest.update(item.read_bytes())
    return digest.hexdigest()


def normalize_settings(raw=None):
    settings = deepcopy(DEFAULTS)
    if raw is not None and not isinstance(raw, dict):
        raise ValueError('Milestone settings must be an object')
    raw = dict(raw or {})
    for alias, key in (('interval_episodes', 'interval'), ('hands_per_game', 'eval_hands'),
                       ('seed_count', 'eval_seeds')):
        if alias in raw:
            raw[key] = raw.pop(alias)
    unknown = set(raw) - set(settings)
    if unknown:
        raise ValueError('Unknown milestone settings: ' + ', '.join(sorted(unknown)))
    settings.update(raw)
    if type(settings['enabled']) is not bool:
        raise ValueError('Milestone enabled must be true or false')
    for key, maximum in (('interval', 10000000), ('eval_hands', 1000), ('eval_seeds', 20)):
        if type(settings[key]) is not int or not 1 <= settings[key] <= maximum:
            raise ValueError(f'Milestone {key} must be an integer from 1 to {maximum}')
    if settings['table_sizes'] != [4, 5, 6]:
        raise ValueError('Milestone benchmarks always compare 4, 5 and 6 players')
    if type(settings['seed']) is not int or not 0 <= settings['seed'] <= 2147483647:
        raise ValueError('Milestone seed must be a non-negative 32-bit integer')
    return settings


def atomic_json(path, value, immutable=False):
    """Publish a whole file; immutable artifacts can never replace old bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        if immutable:
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError('Immutable milestone artifact already differs: ' + str(path))
        else:
            temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def immutable_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        with temporary_path.open('rb') as saved:
            os.fsync(saved.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            if file_hash(source) != file_hash(destination):
                raise ValueError('Immutable checkpoint already differs: ' + str(destination))
    finally:
        temporary_path.unlink(missing_ok=True)


def ensure_settings(folder, request, raw=None):
    """Create a separate immutable sidecar, keeping legacy request hashes intact."""
    if raw is not None and not isinstance(raw, dict):
        raise ValueError('Milestone settings must be an object')
    folder = Path(folder)
    path = folder / 'milestone-settings.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['request_hash'] != request_hash(request):
            raise ValueError('Milestone settings belong to a different training request')
        normalize_settings(saved['settings'])
        if raw is not None:
            proposed = normalize_settings({'seed': request['config'].get('seed', 42), **raw})
            if proposed != saved['settings']:
                raise ValueError('Milestone settings are immutable after the job starts')
        return saved
    settings = normalize_settings({'seed': request['config'].get('seed', 42), **(raw or {})})
    opponents = []
    for opponent in request['opponents']:
        opponents.append({**opponent, 'content_sha256': opponent_hash(opponent['path'])})
    saved = {'schema_version': 1, 'request_hash': request_hash(request), 'settings': settings,
             'source_checkpoint': deepcopy(request['source_checkpoint']), 'opponents': opponents}
    atomic_json(path, saved, immutable=True)
    return saved


@contextmanager
def preserve_rng():
    """Benchmarking must not change the random stream of the next PPO update."""
    import torch
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        # Benchmark inference and model construction use CPU exclusively, so
        # they never initialize or advance the separate MPS random stream.


def next_rollout_count(episodes, target, rollout_count, interval):
    next_boundary = (episodes // interval + 1) * interval
    return min(rollout_count, target - episodes, next_boundary - episodes)


def begin_tracking(folder, request, episodes):
    folder = Path(folder)
    sidecar = ensure_settings(folder, request)
    if not sidecar['settings']['enabled']:
        return None
    for opponent in sidecar['opponents']:
        if opponent_hash(opponent['path']) != opponent['content_sha256']:
            raise ValueError('A frozen milestone opponent checksum changed: ' + opponent['name'])
    path = folder / 'milestone-tracking.json'
    if not path.exists():
        interval = sidecar['settings']['interval']
        atomic_json(path, {'schema_version': 1, 'request_hash': request_hash(request),
                          'settings_sha256': file_hash(folder / 'milestone-settings.json'),
                          'tracking_start_episodes': episodes,
                          'unrecoverable_past_milestones': list(range(interval, episodes, interval)),
                          'note': 'Tracking starts at the actual saved episode count. Earlier policies cannot be reconstructed.'},
                    immutable=True)
    tracking = json.loads(path.read_text())
    if (tracking['request_hash'] != request_hash(request)
            or tracking['settings_sha256'] != file_hash(folder / 'milestone-settings.json')):
        raise ValueError('Milestone tracking checksum does not match its request/settings')
    return sidecar


def archive(folder, request, checkpoint_path, kind, episodes, iteration):
    """Copy a fully committed checkpoint before publishing its manifest."""
    import torch
    from arena.bc_train_worker import build_model, model_layers
    identifier = ('bc' if kind == 'baseline' else 'episode' if kind == 'milestone' else 'final')
    identifier += f'-{episodes:09d}'
    destination = Path(folder) / 'milestones' / identifier
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        return snapshot(folder, identifier)
    destination.mkdir(parents=True, exist_ok=True)
    immutable_copy(Path(checkpoint_path), destination / 'checkpoint.pt')
    saved = torch.load(destination / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if kind == 'baseline':
        if file_hash(destination / 'checkpoint.pt') != request['source_checkpoint']['sha256']:
            raise ValueError('Baseline archive does not match the frozen BC source checksum')
    else:
        if (saved['request_hash'] != request_hash(request) or saved['progress']['episodes'] != episodes
                or saved['feature_version'] != FEATURE_VERSION):
            raise ValueError('Milestone archive must use the exact completed-iteration checkpoint')
    actor_state = saved['model'] if kind == 'baseline' else saved['actor']
    feature_count = int(actor_state['0.weight'].shape[1])
    if feature_count == FEATURE_COUNT:
        actor = build_model()
    else:
        from arena.ppo_context import PLACEMENT_FEATURE_COUNT, CONTEXT_VERSION
        from arena.ppo_models import build_actor
        if feature_count != PLACEMENT_FEATURE_COUNT or saved.get('context_version') != CONTEXT_VERSION:
            raise ValueError('The milestone actor has incompatible game context')
        actor = build_actor(context=True)
    actor.load_state_dict(saved['model'] if kind == 'baseline' else saved['actor'])
    weights = {'schema_version': 1, 'feature_version': FEATURE_VERSION, 'feature_count': feature_count,
               'action_names': ACTION_NAMES, 'layers': model_layers(actor),
               'best_iteration': iteration, 'snapshot_id': identifier, 'episodes': episodes}
    benchmark_reward_config = reward_settings(request['config'])
    if feature_count != FEATURE_COUNT:
        weights.update(context_version=CONTEXT_VERSION, reward_mode='game_placement',
                       reward_config=benchmark_reward_config, selection_metric='placement_chip_score')
    atomic_json(destination / 'weights.json', weights, immutable=True)
    manifest = {'schema_version': 1, 'id': identifier, 'kind': kind,
                'feature_version': FEATURE_VERSION, 'feature_count': feature_count,
                'episodes': episodes, 'iteration': iteration, 'resumable': kind != 'baseline',
                'checkpoint_format': 'bc' if kind == 'baseline' else saved.get('algorithm', 'ppo'),
                'request_hash': request_hash(request),
                'settings_sha256': file_hash(Path(folder) / 'milestone-settings.json'),
                'source_checkpoint_sha256': request['source_checkpoint']['sha256'],
                'checkpoint_sha256': file_hash(destination / 'checkpoint.pt'),
                'weights_sha256': file_hash(destination / 'weights.json')}
    if benchmark_reward_config['reward_mode'] == 'game_placement':
        manifest.update(schema_version=2, benchmark_reward_config=benchmark_reward_config,
                        reward_mode='chips' if kind == 'baseline' else 'game_placement',
                        selection_metric='placement_chip_score')
    if feature_count != FEATURE_COUNT:
        manifest.update(context_version=CONTEXT_VERSION, reward_config=benchmark_reward_config)
    atomic_json(manifest_path, manifest, immutable=True)
    return snapshot(folder, identifier)


def snapshot(folder, snapshot_id):
    if not isinstance(snapshot_id, str) or not SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError('Invalid milestone snapshot identifier')
    folder = Path(folder)
    destination = folder / 'milestones' / snapshot_id
    if destination.resolve() != folder.resolve() / 'milestones' / snapshot_id:
        raise ValueError('Milestone snapshot must stay inside its job directory')
    manifest = json.loads((destination / 'manifest.json').read_text())
    request = json.loads((folder / 'request.json').read_text())
    if (manifest['id'] != snapshot_id or manifest['request_hash'] != request_hash(request)
            or manifest['settings_sha256'] != file_hash(folder / 'milestone-settings.json')
            or manifest['source_checkpoint_sha256'] != request['source_checkpoint']['sha256']):
        raise ValueError('Milestone snapshot request/settings checksum mismatch')
    for name in ('checkpoint', 'weights'):
        path = destination / ('checkpoint.pt' if name == 'checkpoint' else 'weights.json')
        if path.is_symlink() or file_hash(path) != manifest[name + '_sha256']:
            raise ValueError('Milestone snapshot checksum changed: ' + name)
    report_path = destination / 'report.json'
    if report_path.exists():
        checksum_path = destination / 'report-checksum.json'
        if checksum_path.exists() and json.loads(checksum_path.read_text())['sha256'] != file_hash(report_path):
            raise ValueError('Milestone report checksum changed')
    return {**manifest, 'checkpoint_path': str(destination / 'checkpoint.pt'),
            'weights_path': str(destination / 'weights.json'),
            'report_path': str(report_path) if report_path.exists() else None}


def game_plan(settings, opponents):
    from arena.ppo_rollout import selected_opponents
    plan = []
    for group in range(settings['eval_seeds']):
        for table_size in settings['table_sizes']:
            seed = f"ppo:milestone:{settings['seed']}:group:{group}:size:{table_size}"
            lineup = selected_opponents(opponents, table_size - 1, seed + ':lineup')
            for offset in range(table_size):
                plan.append({'game_id': f'size-{table_size}-seed-{group:02d}-seat-{offset}',
                             'seed_group': group, 'seed': seed, 'table_size': table_size,
                             'offset': offset, 'requested_hands': settings['eval_hands'], 'opponents': lineup})
    return plan


def play_benchmark_game(weights, game, work_dir, check_pause=None):
    """Use exported inference and SDK failure handling; never accept a fake score."""
    from macpoker.actions import Action
    from macpoker.match import MatchConfig, MatchRunner
    from macpoker.transport import BotDied, Transport
    from arena.ppo_rollout import OpponentTransport

    class ExportedLearner(Transport):
        def __init__(self):
            self.generator = random.Random(f"{game['seed']}:learner:{game['offset']}")
            self.context = None
            if weights.get('context_version') is not None:
                from arena.ppo_context import GameContext, CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT
                if weights['context_version'] != CONTEXT_VERSION or weights.get('feature_count') != PLACEMENT_FEATURE_COUNT:
                    raise ValueError('The milestone weights use incompatible game context')
                self.context = GameContext()

        def send(self, message):
            if check_pause is not None:
                check_pause()
            if self.context is not None:
                self.context.send(message)

        def act(self, observation, timeout_ms):
            if check_pause is not None:
                check_pause()
            started = time.perf_counter()
            try:
                features = self.context.encode(observation) if self.context is not None else encode_observation(observation)
                logits = forward_logits(features, weights['layers'])
                if len(logits) != len(ACTION_NAMES) or not all(math.isfinite(value) for value in logits):
                    raise ValueError('Exported learner produced invalid logits')
                action = choose_action(observation, logits, self.generator)
                return Action.from_wire(action), (time.perf_counter() - started) * 1000.0
            except Exception as error:
                raise BotDied(str(error)) from error

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    transports = [ExportedLearner()]
    runner = None
    runner_path = Path(__file__).with_name('bot_runner.py')
    with tempfile.TemporaryDirectory(prefix='milestone-game-', dir=work_dir) as temporary:
        try:
            for player, opponent in enumerate(game['opponents'], start=1):
                source = Path(opponent['path'])
                source_folder = source.parent if source.is_file() else source
                destination = Path(temporary) / f'opponent-{player}'
                shutil.copytree(source_folder, destination)
                command = [sys.executable, '-B', '-u', str(runner_path), str(destination / 'main.py'),
                           f"{game['seed']}:opponent:{player}"]
                transports.append(OpponentTransport(command))
            config = MatchConfig(seats=game['table_size'], deals=game['requested_hands'],
                                 offset=game['offset'], seed=game['seed'], stack=200, sb=1, bb=2)
            runner = MatchRunner(config, transports)
            result = runner.run()
            completed_hands = len(result.hands)
            valid_result = (len(result.chips) == game['table_size']
                            and len(result.verdicts) == game['table_size']
                            and result.num_hands == completed_hands
                            and sum(result.chips) == 0
                            and all(type(chips) in (int, float) and math.isfinite(chips) for chips in result.chips))
            complete = valid_result and completed_hands == game['requested_hands']
            placement = points(result.chips) if complete else None
            return {'completed_hands': completed_hands, 'completed': complete,
                    'chips': list(result.chips), 'verdicts': list(result.verdicts),
                    'points': placement, 'rank': game['table_size'] + 1 - placement[0] if placement else None,
                    'normalized_placement': (placement[0] - 1) / (game['table_size'] - 1) if placement else None,
                    'net_chips': result.chips[0] if complete else None,
                    'bb100': result.chips[0] / 2.0 / completed_hands * 100 if complete else None,
                    'error': None if complete else 'Official SDK result was incomplete or inconsistent'}
        except Exception as error:
            from arena.ppo_train_worker import TrainingPaused
            if isinstance(error, TrainingPaused):
                error.benchmark_completed_hands = len(runner.hands) if runner is not None else 0
                raise
            return {'completed_hands': len(runner.hands) if runner is not None else 0,
                    'completed': False, 'chips': list(runner.chips) if runner is not None else None,
                    'verdicts': list(runner.verdicts) if runner is not None else None,
                    'points': None, 'rank': None, 'normalized_placement': None,
                    'net_chips': None, 'bb100': None, 'error': str(error)}
        finally:
            for transport in transports:
                transport.close()


def aggregate_games(games, expected_games):
    completed = [game for game in games if game['completed']]
    learner_failures = sum(bool(game['verdicts']) and game['verdicts'][0] != 'OK' for game in games)
    opponent_failures = sum(sum(verdict != 'OK' for verdict in game['verdicts'][1:])
                            for game in games if game['verdicts'])
    failed_games = sum(bool(game['error']) or not game['completed']
                       or any(verdict != 'OK' for verdict in game['verdicts'] or []) for game in games)
    hands = sum(game['completed_hands'] for game in completed)
    chips = sum(game['net_chips'] for game in completed) if completed else None
    return {'expected_games': expected_games, 'attempted_games': len(games),
            'completed_games': len(completed), 'attempted_hands': sum(game['requested_hands'] for game in games),
            'completed_hands': sum(game['completed_hands'] for game in games),
            'scored_hands': hands, 'failed_games': failed_games,
            'learner_failures': learner_failures, 'opponent_failures': opponent_failures,
            'net_chips': chips, 'bb100': chips / 2.0 / hands * 100 if hands else None,
            'mean_points': sum(game['points'][0] for game in completed) / len(completed) if completed else None,
            'mean_rank': sum(game['rank'] for game in completed) / len(completed) if completed else None,
            'normalized_placement': sum(game['normalized_placement'] for game in completed) / len(completed) if completed else None,
            'eligible_for_best': len(completed) == expected_games and failed_games == 0}


def report_from_games(manifest, settings, games, plan):
    result = aggregate_games(games, len(plan))
    by_size = {}
    for size in settings['table_sizes']:
        selected = [game for game in games if game['table_size'] == size]
        by_size[str(size)] = aggregate_games(selected, size * settings['eval_seeds'])
    # Equal table-size weights prevent the six-player schedule dominating.
    for metric in ('bb100', 'normalized_placement'):
        values = [row[metric] for row in by_size.values()]
        result[metric] = sum(values) / len(values) if all(value is not None for value in values) else None
    complete = len(games) == len(plan)
    report = {'schema_version': 1, 'snapshot_id': manifest['id'], 'request_hash': manifest['request_hash'],
            'settings_sha256': manifest['settings_sha256'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
            'weights_sha256': manifest['weights_sha256'], 'settings': settings,
            'status': 'completed' if complete and result['eligible_for_best'] else 'failed' if complete else 'pending',
            **result, 'by_table_size': by_size, 'games': games,
            'aggregation': 'Equal mean of the 4/5/6-player bb100 and normalized placement scores. Raw points and rank are game means.',
            'inference': 'Serialized weights, shared exported forward_logits/choose_action, fixed Python random seed',
            'failure_policy': 'SDK check-fold continuation is reported. Any failed or incomplete scheduled game excludes this snapshot from both leaders.'}
    if manifest.get('schema_version', 1) >= 2:
        objective = manifest['benchmark_reward_config']
        rounds = aggregate_rounds(games, placement_weight=objective['placement_weight'],
                                  chip_scale_bb100=objective['chip_scale_bb100'],
                                  expected_seeds=settings['eval_seeds'], table_sizes=settings['table_sizes'])
        report.update(schema_version=2, round_metrics=rounds, selection_score=rounds['selection_score'],
                      selection_metric=rounds['selection_metric'], reward_config=objective,
                      normalized_round_placement=rounds['normalized_round_placement'])
        report['eligible_for_best'] = report['eligible_for_best'] and rounds['eligible_for_best']
    return report


def evaluate_snapshot(folder, manifest, sidecar, check_pause=None, on_progress=None):
    folder = Path(folder)
    destination = folder / 'milestones' / manifest['id']
    report_path = destination / 'report.json'
    checksum_path = destination / 'report-checksum.json'
    if report_path.exists() and checksum_path.exists():
        if json.loads(checksum_path.read_text())['sha256'] != file_hash(report_path):
            raise ValueError('Milestone report checksum changed')
        return json.loads(report_path.read_text())
    settings = sidecar['settings']
    plan = game_plan(settings, sidecar['opponents'])
    weights = json.loads((destination / 'weights.json').read_text())
    games = []
    for game in plan:
        if check_pause is not None:
            check_pause()
        game_path = destination / 'games' / (game['game_id'] + '.json')
        if game_path.exists():
            saved = json.loads(game_path.read_text())
            record = {key: value for key, value in saved.items() if key != 'record_sha256'}
            if (saved['checkpoint_sha256'] != manifest['checkpoint_sha256']
                    or saved['settings_sha256'] != manifest['settings_sha256']
                    or saved.get('record_sha256') != content_hash(record)
                    or any(saved.get(key) != value for key, value in game.items())):
                raise ValueError('Saved benchmark game checksum mismatch')
        else:
            active = {**game, 'checkpoint_sha256': manifest['checkpoint_sha256'],
                      'settings_sha256': manifest['settings_sha256']}
            atomic_json(destination / 'active-game.json', active)
            try:
                result = play_benchmark_game(weights, game, folder / 'milestone-work', check_pause)
            except Exception as error:
                from arena.ppo_train_worker import TrainingPaused
                if isinstance(error, TrainingPaused):
                    active.update(status='paused', completed_hands=getattr(error, 'benchmark_completed_hands', 0))
                    atomic_json(destination / 'active-game.json', active)
                raise
            saved = {**active, **result}
            saved['record_sha256'] = content_hash(saved)
            atomic_json(game_path, saved, immutable=True)
            (destination / 'active-game.json').unlink(missing_ok=True)
        games.append(saved)
        partial = report_from_games(manifest, settings, games, plan)
        atomic_json(destination / 'progress.json', partial)
        if on_progress is not None:
            on_progress(partial)
    report = report_from_games(manifest, settings, games, plan)
    report['game_checksums'] = {game['game_id']: file_hash(destination / 'games' / (game['game_id'] + '.json')) for game in games}
    atomic_json(report_path, report, immutable=True)
    atomic_json(checksum_path, {'sha256': file_hash(report_path)}, immutable=True)
    return report


def checkpoint_hook(folder, request, progress, sidecar, check_pause=None, on_progress=None):
    """Run only between complete PPO updates, including on resume."""
    folder = Path(folder)
    settings = sidecar['settings']
    with preserve_rng():
        baseline = archive(folder, request, request['source_checkpoint']['path'], 'baseline', 0, 0)
        evaluate_snapshot(folder, baseline, sidecar, check_pause, on_progress)
        episodes = progress['episodes']
        kind = None
        if episodes and episodes % settings['interval'] == 0:
            kind = 'milestone'
        elif episodes and episodes == request['config']['target_episodes']:
            kind = 'final'
        if kind is not None:
            manifest = archive(folder, request, folder / 'latest.pt', kind, episodes, progress['iteration'])
            evaluate_snapshot(folder, manifest, sidecar, check_pause, on_progress)
    return summaries(folder)


def summaries(folder):
    """Compact dashboard data; short PPO validation scores never enter leaders."""
    folder = Path(folder)
    settings_path = folder / 'milestone-settings.json'
    if not settings_path.exists():
        return {'enabled': False, 'settings': None, 'tracking_start_episodes': None,
                'baseline': None, 'snapshots': [], 'leaders': {'bb100': None, 'normalized_placement': None},
                'round_leaders': {'placement_chip_score': None, 'normalized_round_placement': None}, 'pending': []}
    sidecar = json.loads(settings_path.read_text())
    result = {'enabled': sidecar['settings']['enabled'], 'settings': sidecar['settings'],
              'tracking_start_episodes': None, 'unrecoverable_past_milestones': [],
              'baseline': None, 'snapshots': [], 'leaders': {'bb100': None, 'normalized_placement': None},
              'round_leaders': {'placement_chip_score': None, 'normalized_round_placement': None}, 'pending': []}
    tracking_path = folder / 'milestone-tracking.json'
    if tracking_path.exists():
        tracking = json.loads(tracking_path.read_text())
        result.update(tracking_start_episodes=tracking['tracking_start_episodes'],
                      unrecoverable_past_milestones=tracking['unrecoverable_past_milestones'])
    candidates = []
    for path in sorted((folder / 'milestones').glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        destination = path.parent
        summary = {**manifest, 'checkpoint_available': (destination / 'checkpoint.pt').is_file(),
                   'status': 'pending', 'eligible_for_best': False,
                   'bb100': None, 'normalized_placement': None, 'by_table_size': {},
                   'attempted_games': 0, 'completed_games': 0, 'attempted_hands': 0,
                   'completed_hands': 0, 'learner_failures': 0, 'opponent_failures': 0}
        report_path = destination / 'report.json'
        progress_path = report_path if report_path.exists() else destination / 'progress.json'
        if progress_path.exists():
            report = json.loads(progress_path.read_text())
            if report['checkpoint_sha256'] != manifest['checkpoint_sha256'] or report['settings_sha256'] != manifest['settings_sha256']:
                raise ValueError('Milestone report checksum linkage changed')
            if report_path.exists():
                checksum_path = destination / 'report-checksum.json'
                if checksum_path.exists() and json.loads(checksum_path.read_text())['sha256'] != file_hash(report_path):
                    raise ValueError('Milestone report checksum changed')
            summary.update({key: value for key, value in report.items()
                            if key not in ('games', 'game_checksums', 'settings', 'schema_version')})
            if not report_path.exists() or not (destination / 'report-checksum.json').exists():
                summary.update(status='pending', eligible_for_best=False)
            # Derive new comparisons in memory for old immutable report files.
            # Individual-game placement retains its original meaning.
            if summary['status'] != 'pending':
                objective = manifest.get('benchmark_reward_config', reward_settings({}))
                round_metrics = report.get('round_metrics')
                if round_metrics is None:
                    round_metrics = aggregate_rounds(report['games'],
                        placement_weight=objective['placement_weight'], chip_scale_bb100=objective['chip_scale_bb100'],
                        expected_seeds=sidecar['settings']['eval_seeds'], table_sizes=sidecar['settings']['table_sizes'])
                summary.update(round_metrics=round_metrics, round_selection_score=round_metrics['selection_score'],
                               normalized_round_placement=round_metrics['normalized_round_placement'])
        active_path = destination / 'active-game.json'
        if summary['status'] == 'pending' and active_path.exists():
            active = json.loads(active_path.read_text())
            # A paused game restarts from its fixed seed. These are distinct
            # scheduled game slots, so a retried slot is never counted twice.
            if not (destination / 'games' / (active['game_id'] + '.json')).exists():
                summary['attempted_games'] += 1
                summary['attempted_hands'] += active['requested_hands']
                summary['completed_hands'] += active.get('completed_hands', 0)
                summary['active_game'] = active['game_id']
        if summary['status'] == 'pending':
            result['pending'].append(manifest['id'])
        if manifest['kind'] == 'baseline':
            result['baseline'] = summary
        else:
            result['snapshots'].append(summary)
        if summary['eligible_for_best']:
            candidates.append(summary)
    for metric in result['leaders']:
        eligible = [candidate for candidate in candidates if candidate.get(metric) is not None]
        if eligible:
            result['leaders'][metric] = max(eligible, key=lambda row: (row[metric], -row['episodes']))['id']
    for metric, field in (('placement_chip_score', 'round_selection_score'),
                          ('normalized_round_placement', 'normalized_round_placement')):
        eligible = [candidate for candidate in candidates if candidate.get(field) is not None
                    and candidate.get('round_metrics', {}).get('eligible_for_best')]
        if eligible:
            result['round_leaders'][metric] = max(eligible, key=lambda row: (row[field], -row['episodes']))['id']
    if any(candidate.get('selection_metric') == 'placement_chip_score' for candidate in candidates):
        result['leaders']['placement_chip_score'] = result['round_leaders']['placement_chip_score']
    return result
