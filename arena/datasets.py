"""Collect resumable poker demonstrations into immutable, compressed datasets."""

from collections import Counter
import gzip
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import shutil
import time
import threading
import uuid
import zipfile

from arena import study

ARCHIVE_LOCK = threading.Lock()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def checked_name(body):
    name = str(body.get('name', '')).strip()
    if not name or len(name) > 100:
        raise ValueError('Give the dataset a name of 1–100 characters')
    return name


def queue(server, body):
    name = checked_name(body)
    if 'target_episodes' in body and 'target_records' in body:
        raise ValueError('Choose an episode target or a legacy decision target, not both')
    # Existing API clients and saved collections retain their decision target.
    target_unit = 'decisions' if 'target_records' in body else 'episodes'
    target = body['target_records'] if target_unit == 'decisions' else body.get('target_episodes', 100000)
    if type(target) is not int or not 1 <= target <= 100000000:
        raise ValueError(f'Target {target_unit} must be an integer from 1 to 100,000,000')
    disk_limit = body.get('max_disk_gb', 10)
    if type(disk_limit) not in (int, float) or not math.isfinite(disk_limit) or not 0.1 <= disk_limit <= 1000:
        raise ValueError('Dataset storage limit must be between 0.1 and 1,000 GB')
    settings = dict(body, seeds=['collection-preview'], batches=1)
    plan = server.study_preview(settings)
    teachers = body.get('teachers', [])
    agents = body['agents']
    if (not isinstance(teachers, list) or not teachers or
            not all(isinstance(identifier, str) for identifier in teachers) or
            len(set(teachers)) != len(teachers)):
        raise ValueError('Choose at least one distinct teacher')
    if any(identifier not in agents for identifier in teachers):
        raise ValueError('Teachers must also be selected players')
    prefix = str(body.get('seed_prefix', 'collection')).strip()
    if not prefix or len(prefix) > 80:
        raise ValueError('Seed prefix must contain 1–80 characters')
    minimum_groups = body.get('minimum_seed_groups', 3)
    if type(minimum_groups) is not int or not 3 <= minimum_groups <= 1000:
        raise ValueError('Collect at least 3 and at most 1,000 minimum seed groups')
    config = {
        'agents': agents, 'teachers': teachers, 'table_sizes': sorted(body.get('table_sizes', [4, 5, 6])),
        'deals': body.get('deals', 100), 'seed_prefix': prefix,
        'max_disk_gb': disk_limit, 'minimum_seed_groups': minimum_groups,
    }
    dataset = new_dataset(server, name, teachers, target, config, target_unit=target_unit)
    if target_unit == 'episodes':
        episodes_per_batch = plan['hands']
        planned_batches = max(minimum_groups, math.ceil(target / episodes_per_batch))
        dataset['planned_episodes'] = planned_batches * episodes_per_batch
        server.save('datasets', dataset)
    return dataset


def new_dataset(server, name, teacher_ids, target, config, source_run_id=None, target_unit='decisions', persist=True):
    teachers = []
    for identifier in teacher_ids:
        agent = server.get('agents', identifier)
        teachers.append({field: agent[field] for field in ('id', 'name', 'hash')})
    dataset = {
        'id': uuid.uuid4().hex, 'name': name, 'status': 'queued', 'created': server.now(),
        'kind': 'import' if source_run_id else 'collection', 'source_run_id': source_run_id,
        'config': config, 'target_unit': target_unit,
        'target_episodes': target if target_unit == 'episodes' else None,
        'target_records': target if target_unit == 'decisions' else None,
        'episodes': 0, 'records': 0, 'bytes': 0,
        'teachers': teachers, 'teacher_counts': {teacher['id']: 0 for teacher in teachers},
        'table_size_counts': {}, 'teacher_size_counts': {}, 'street_counts': {}, 'action_counts': {},
        'seed_groups': [], 'games': 0, 'completed_blocks': 0, 'block_game': 0,
        'excluded_records': 0, 'failures': 0, 'empty_blocks': 0,
        'elapsed_seconds': 0, 'episodes_per_second': 0, 'records_per_second': 0, 'pause_requested': False,
        'finish_requested': False, 'progress': 'Waiting in the local work queue', 'error': None,
        'engine_config': {'engine': 'macpoker', 'version': version('macpoker'), 'stack': 200,
                          'blinds': [1, 2], 'bank_ms': 30000, 'increment_ms': 100, 'schedule_version': 1},
    }
    (server.DATA / 'datasets' / dataset['id']).mkdir(parents=True)
    if persist:
        server.save('datasets', dataset)
    return dataset


def import_run(server, body):
    name = checked_name(body)
    run = server.get('runs', body.get('run_id', ''))
    if run['status'] != 'completed' or not run.get('training', {}).get('records'):
        raise ValueError('Choose a completed run with recorded teacher decisions')
    if any(dataset.get('source_run_id') == run['id'] for dataset in server.all_rows('datasets')):
        raise ValueError('That recorded run is already in the dataset library')
    teachers = [teacher['id'] for teacher in run['training']['teachers']]
    dataset = new_dataset(server, name, teachers, run['training']['records'], {}, run['id'])
    dataset['engine_config'] = dict(run.get('profile_config', dataset['engine_config']))
    server.save('datasets', dataset)
    return dataset


def control(server, key, action, body=None):
    with server.LOCK:
        dataset = server.get('datasets', key)
        status = dataset['status']
        if action == 'pause':
            if status not in ('queued', 'running'):
                raise ValueError('Only queued or running collections can be paused')
            if status == 'queued':
                dataset.update(status='paused', progress='Paused before starting')
            else:
                server.PAUSE.add(key)
                dataset['pause_requested'] = True
        elif action == 'resume':
            if status not in ('paused', 'interrupted', 'failed'):
                raise ValueError('Only paused, interrupted or failed collections can resume')
            if body and 'max_disk_gb' in body and dataset['kind'] == 'collection':
                limit = body['max_disk_gb']
                if type(limit) not in (int, float) or not math.isfinite(limit) or not 0.1 <= limit <= 1000:
                    raise ValueError('Storage limit must be between 0.1 and 1,000 GB')
                dataset['config']['max_disk_gb'] = limit
            server.PAUSE.discard(key)
            dataset.update(status='queued', pause_requested=False, error=None, progress='Queued to continue saved collection')
        elif action == 'finish':
            if dataset['kind'] == 'import':
                raise ValueError('Imported runs finish after all their files are copied')
            if status == 'ready':
                return dataset
            if status not in ('queued', 'running', 'paused', 'interrupted', 'failed'):
                raise ValueError('This dataset cannot be finalized')
            dataset['finish_requested'] = True
            if status != 'running':
                server.PAUSE.discard(key)
                dataset.update(status='queued', pause_requested=False, error=None)
            dataset['progress'] = ('Will finish after the current PHH hand' if dataset['kind'] == 'phh'
                                   else 'Will finish after the current balanced batch')
        else:
            raise ValueError('Unknown dataset action')
        server.save('datasets', dataset)
        return dataset


def shards(server, key):
    with server.connect() as database:
        rows = database.execute('SELECT payload FROM dataset_shards WHERE id LIKE ? ORDER BY id', (key + '-%',)).fetchall()
    return [json.loads(row[0]) for row in rows]


def detail(server, key):
    dataset = server.get('datasets', key)
    restore_episode_count(server, dataset)
    dataset['folder'] = str(server.DATA / 'datasets' / key)
    dataset['seed_group_count'] = len(dataset['seed_groups'])
    if dataset['status'] == 'ready':
        dataset['download_url'] = f'/api/datasets/{key}/download'
    return dataset


def restore_episode_count(server, dataset):
    """Read older checkpoints without changing their target or frozen files."""
    dataset.setdefault('target_unit', 'episodes' if dataset.get('target_episodes') else 'decisions')
    if 'episodes' not in dataset:
        if dataset['kind'] == 'collection':
            dataset['episodes'] = dataset['games'] * dataset['config']['deals']
        else:
            try:
                run = server.get('runs', dataset['source_run_id'])
            except ValueError:
                if dataset['status'] != 'ready':
                    raise
                # A frozen import is usable even if its original run is gone.
                dataset.update(episodes=None, episodes_per_second=None)
                return
            imported_games = []
            for table in run['results']:
                for result in table['games']:
                    if result.get('training'):
                        imported_games.append(result)
            dataset['episodes'] = sum(result['num_hands'] for result in imported_games[:dataset['games']])
    dataset['episodes_per_second'] = dataset['episodes'] / max(0.001, dataset['elapsed_seconds'])


def save_progress(server, dataset):
    # HTTP controls and worker checkpoints share a lock. Preserve requests that
    # arrived while the worker was processing a game.
    with server.LOCK:
        current = server.get('datasets', dataset['id'])
        if dataset['status'] != 'ready':
            dataset['finish_requested'] = current.get('finish_requested', False)
        dataset['pause_requested'] = dataset['id'] in server.PAUSE
        server.save('datasets', dataset)


def write_shard(server, dataset, source, number, table_size):
    """Compress one finished game, validating records without retaining them."""
    folder = server.DATA / 'datasets' / dataset['id']
    name = f'part-{number:08d}.jsonl.gz'
    destination = folder / name
    temporary = destination.with_suffix('.tmp')
    teachers = {teacher['id']: teacher for teacher in dataset['teachers']}
    teacher_counts = Counter()
    size_counts = Counter()
    street_counts = Counter()
    action_counts = Counter()
    teacher_size_counts = {}
    seed_groups = set()
    records = 0
    try:
        with source.open(encoding='utf-8') as stream, gzip.open(temporary, 'wt', encoding='utf-8') as output:
            for line in stream:
                record = json.loads(line)
                teacher = record['teacher']
                expected = teachers.get(teacher['id'])
                if expected is None or expected['hash'] != teacher['hash'] or record.get('verdict') != 'OK':
                    raise ValueError('Decision contains an unexpected teacher version or failure')
                if not record.get('seed_group') or not record.get('decision_id'):
                    raise ValueError('Decision is missing its seed group or identity')
                observation = record['observation']
                if not isinstance(observation, dict) or record['executed_action']['action'] not in ('fold', 'check', 'call', 'raise'):
                    raise ValueError('Decision has an unsupported observation or action')
                size = int(record.get('table_size', table_size))
                if size != table_size:
                    raise ValueError('Decision table size does not match its game')
                record['table_size'] = size
                output.write(json.dumps(record, separators=(',', ':')) + '\n')
                records += 1
                teacher_counts[teacher['id']] += 1
                size_counts[str(size)] += 1
                street_counts[str(observation.get('street', 'unknown'))] += 1
                action_counts[record['executed_action']['action']] += 1
                teacher_size_counts.setdefault(teacher['id'], Counter())[str(size)] += 1
                seed_groups.add(record['seed_group'])
        with temporary.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if dataset['kind'] == 'collection' and dataset['bytes'] + temporary.stat().st_size > dataset['config']['max_disk_gb'] * 1024 ** 3:
            raise ValueError('Dataset storage limit reached. Saved decisions are preserved; increase the limit before resuming.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        'id': f"{dataset['id']}-{number:08d}", 'name': name, 'sha256': digest,
        'records': records, 'bytes': destination.stat().st_size, 'teacher_counts': dict(teacher_counts),
        'table_size_counts': dict(size_counts), 'teacher_size_counts': teacher_size_counts,
        'street_counts': dict(street_counts), 'action_counts': dict(action_counts), 'seed_groups': sorted(seed_groups),
    }


def commit_shard(server, dataset, shard, result):
    """Commit the file index, counters and next-game cursor in one transaction."""
    episodes = result['num_hands']
    if type(episodes) is not int or episodes < 0:
        raise ValueError('A completed game must report a non-negative integer hand count')
    with server.LOCK, server.connect() as database:
        if database.execute('SELECT 1 FROM dataset_shards WHERE id=?', (shard['id'],)).fetchone():
            raise ValueError('This game has already been committed; resume from the saved checkpoint')
        current = server.get('datasets', dataset['id'])
        dataset['finish_requested'] = current.get('finish_requested', False)
        dataset['pause_requested'] = dataset['id'] in server.PAUSE
        dataset['episodes'] += episodes
        dataset['records'] += shard['records']
        dataset['bytes'] += shard['bytes']
        dataset['games'] += 1
        dataset['block_game'] += 1
        dataset['excluded_records'] += result.get('training', {}).get('excluded_records', 0)
        dataset['failures'] += sum(verdict != 'OK' for verdict in result.get('verdicts', []))
        for field in ('teacher_counts', 'table_size_counts', 'street_counts', 'action_counts'):
            for key, count in shard[field].items():
                dataset[field][key] = dataset[field].get(key, 0) + count
        for teacher, counts in shard['teacher_size_counts'].items():
            existing = dataset['teacher_size_counts'].setdefault(teacher, {})
            for size, count in counts.items():
                existing[size] = existing.get(size, 0) + count
        dataset['seed_groups'] = sorted(set(dataset['seed_groups']) | set(shard['seed_groups']))
        dataset['records_per_second'] = dataset['records'] / max(0.001, dataset['elapsed_seconds'])
        dataset['episodes_per_second'] = dataset['episodes'] / max(0.001, dataset['elapsed_seconds'])
        shard['episodes'] = episodes
        database.execute('INSERT INTO dataset_shards(id,payload) VALUES (?,?)', (shard['id'], json.dumps(shard)))
        database.execute('UPDATE datasets SET payload=? WHERE id=?', (json.dumps(dataset), dataset['id']))


def seal(server, dataset):
    if not dataset['records']:
        raise ValueError('No valid decisions were recorded; check the selected teachers and their errors')
    folder = server.DATA / 'datasets' / dataset['id']
    files = shards(server, dataset['id'])
    manifest = {
        'schema_version': 1, 'id': dataset['id'], 'name': dataset['name'], 'created': dataset['created'],
        'teachers': dataset['teachers'], 'config': dataset['config'], 'source_run_id': dataset['source_run_id'],
        'engine_config': dataset.get('engine_config', {}),
        'episode_definition': 'One completed poker hand at one table, counted once for the whole table. Seat rotations count as separate played hands.',
        'episodes': dataset['episodes'], 'target_unit': dataset['target_unit'],
        'target_episodes': dataset.get('target_episodes'), 'target_records': dataset.get('target_records'),
        'records': dataset['records'], 'bytes': dataset['bytes'], 'seed_groups': dataset['seed_groups'],
        'teacher_counts': dataset['teacher_counts'], 'table_size_counts': dataset['table_size_counts'],
        'teacher_size_counts': dataset['teacher_size_counts'], 'street_counts': dataset['street_counts'],
        'action_counts': dataset['action_counts'], 'files': files,
        'usage': 'Use observation fields only as features and executed_action as labels. Keep every seed_group in one split across datasets. Teacher identity is for filtering, not an input feature.',
    }
    if dataset['kind'] == 'phh':
        manifest.update(source='PHH', phh_stats=dataset['phh_stats'],
                        source_fingerprints=dataset.get('source_fingerprints', {}),
                        episode_definition='One imported PHH hand with at least one eligible teacher decision. Skipped source hands are reported separately.')
    atomic_json(folder / 'manifest.json', manifest)
    dataset.update(status='ready', finished=server.now(), progress='Dataset complete and frozen', pause_requested=False, finish_requested=False)
    server.PAUSE.discard(dataset['id'])
    save_progress(server, dataset)


def pause_if_requested(server, dataset):
    if server.STOP.is_set() or dataset['id'] in server.PAUSE:
        dataset.update(status='interrupted' if server.STOP.is_set() else 'paused', progress='Completed games saved; ready to resume')
        server.PAUSE.discard(dataset['id'])
        save_progress(server, dataset)
        return True
    return False


def clean_committed_game(server, key, number):
    # These temporary files were created by this collection. The immutable
    # compressed shard and database checkpoint are already committed.
    folder = server.DATA / 'runs' / key
    for suffix in ('.json', '.request.json', '.decisions.jsonl', '.log'):
        (folder / f'game-{number}{suffix}').unlink(missing_ok=True)


def execute(server, dataset):
    restore_episode_count(server, dataset)
    dataset.update(status='running', started=dataset.get('started', server.now()), error=None)
    save_progress(server, dataset)
    if dataset['kind'] == 'phh':
        from arena import phh_jobs
        return phh_jobs.execute(server, dataset)
    if dataset['kind'] == 'import':
        return execute_import(server, dataset)
    # A shutdown can happen just after committing a shard but before removing
    # its temporary replay. Only sweep files for already committed games.
    working_folder = server.DATA / 'runs' / dataset['id']
    for path in working_folder.glob('game-*'):
        match = re.fullmatch(r'game-(\d+)(\.json|\.request\.json|\.decisions\.jsonl|\.log)', path.name)
        if match and int(match[1]) <= dataset['games']:
            path.unlink(missing_ok=True)
    config = dataset['config']
    agents = {identifier: server.get('agents', identifier) for identifier in config['agents']}
    previous_elapsed = dataset['elapsed_seconds']
    started = time.monotonic()
    while True:
        if pause_if_requested(server, dataset):
            return
        current = server.get('datasets', dataset['id'])
        dataset['finish_requested'] = current['finish_requested']
        if dataset['target_unit'] == 'episodes':
            target_reached = dataset['episodes'] >= dataset['target_episodes']
        else:
            target_reached = dataset['records'] >= dataset['target_records']
        enough = target_reached and len(dataset['seed_groups']) >= config['minimum_seed_groups']
        if dataset['block_game'] == 0 and (enough or dataset['finish_requested']) and dataset['records']:
            seal(server, dataset)
            return
        seed_group = f"{config['seed_prefix']}:{dataset['id']}:block:{dataset['completed_blocks'] + 1}"
        schedule = study.build_schedule(config['agents'], config['table_sizes'], [seed_group])
        run = {'id': dataset['id'], 'mode': 'study', 'schedule': schedule,
               'config': {'deals': config['deals'], 'record_training': True, 'teachers': config['teachers']}}
        game_index = 0
        records_before_block = dataset.get('block_start_records', dataset['records'])
        dataset['block_start_records'] = records_before_block
        for table in schedule:
            participants = [agents[identifier] for identifier in table['agent_ids']]
            for offset in range(table['table_size']):
                game_index += 1
                if game_index <= dataset['block_game']:
                    continue
                if pause_if_requested(server, dataset):
                    return
                if shutil.disk_usage(server.DATA).free < 256 * 1024 ** 2:
                    raise ValueError('Less than 256 MB of disk space remains. Collection stopped with completed games preserved.')
                dataset['progress'] = f"Batch {dataset['completed_blocks'] + 1} · {table['table_size']} players · game {game_index} · simulating hands and recording teacher decisions"
                save_progress(server, dataset)
                number = dataset['games'] + 1
                result = server.game(run, participants, table['seed'], offset, number, seed_group=seed_group)
                source = server.DATA / 'runs' / dataset['id'] / result['training']['file']
                shard = write_shard(server, dataset, source, number, table['table_size'])
                dataset['elapsed_seconds'] = previous_elapsed + time.monotonic() - started
                commit_shard(server, dataset, shard, result)
                clean_committed_game(server, dataset['id'], number)
        dataset['completed_blocks'] += 1
        dataset['block_game'] = 0
        dataset.pop('block_start_records', None)
        dataset['empty_blocks'] = dataset['empty_blocks'] + 1 if dataset['records'] == records_before_block else 0
        save_progress(server, dataset)
        if dataset['empty_blocks'] >= 3:
            raise ValueError('Three batches produced no valid teacher decisions. Check teacher failures before resuming.')


def execute_import(server, dataset):
    run = server.get('runs', dataset['source_run_id'])
    sources = []
    for table in run['results']:
        for result in table['games']:
            if result.get('training'):
                sources.append((result, len(table['agents'])))
    previous_elapsed = dataset['elapsed_seconds']
    started = time.monotonic()
    for index, (result, table_size) in enumerate(sources):
        if index < dataset['games']:
            continue
        if pause_if_requested(server, dataset):
            return
        filename = result['training']['file']
        if not re.fullmatch(r'game-\d+\.decisions\.jsonl', filename):
            raise ValueError('Unexpected decision filename')
        source = server.DATA / 'runs' / run['id'] / filename
        if source.is_symlink():
            raise ValueError('Decision files must not be symbolic links')
        dataset['progress'] = f'Importing recorded game {index + 1}/{len(sources)}'
        shard = write_shard(server, dataset, source, index + 1, table_size)
        dataset['elapsed_seconds'] = previous_elapsed + time.monotonic() - started
        commit_shard(server, dataset, shard, result)
    seal(server, dataset)


def archive(server, key):
    dataset = server.get('datasets', key)
    if dataset['status'] != 'ready':
        raise ValueError('Finish collection before downloading this dataset')
    folder = server.DATA / 'datasets' / key
    destination = folder / 'dataset.zip'
    if destination.exists():
        return destination
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    temporary = folder / 'dataset.zip.tmp'
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as output:
            output.write(folder / 'manifest.json', 'manifest.json')
            for shard in manifest['files']:
                if not re.fullmatch(r'part-\d{8}\.jsonl\.gz', shard['name']):
                    raise ValueError('Unexpected dataset filename')
                path = folder / shard['name']
                if path.is_symlink():
                    raise ValueError('Dataset files must not be symbolic links')
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != shard['sha256']:
                        raise ValueError('Dataset checksum changed; download refused')
                output.write(path, shard['name'])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
