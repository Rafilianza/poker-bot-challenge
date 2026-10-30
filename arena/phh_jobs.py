"""Import local PHH hands into the dashboard's ordinary BC dataset format."""
from collections import Counter, defaultdict
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import time
import tomllib

from arena import datasets

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / 'external' / 'phh-dataset'
MAX_SOURCE_BYTES = 32 * 1024 * 1024
PREVIEW_HANDS = 100
BATCH_HANDS = 100


def settings(body):
    source = Path(str(body.get('source_path', DEFAULT_SOURCE))).expanduser().resolve()
    collection = body.get('collection', 'pluribus')
    if collection not in ('pluribus', 'folder'):
        raise ValueError('Choose Pluribus or a custom PHH folder')
    folder = source / 'data' / 'pluribus' if collection == 'pluribus' else source
    if not folder.is_dir():
        raise ValueError(f'PHH source folder does not exist: {folder}')
    limit = body.get('max_hands', 10000)
    if type(limit) is not int or not 1 <= limit <= 1000000:
        raise ValueError('Hand limit must be an integer from 1 to 1,000,000')
    profitable = body.get('profitable_only', False)
    if type(profitable) is not bool:
        raise ValueError('Profitable hands only must be true or false')
    teacher = str(body.get('teacher_name', 'Pluribus')).strip()
    if len(teacher) > 100:
        raise ValueError('Teacher name must be at most 100 characters')
    group_root = folder
    namespace = 'pluribus' if collection == 'pluribus' else folder.name
    # Selecting an individual Pluribus session must retain the same teacher and
    # split identities as importing that session through the repository root.
    for parent in (folder, *folder.parents):
        if parent.name == 'pluribus' and parent.parent.name == 'data':
            group_root = parent
            namespace = 'pluribus'
            break
    return {'source_path': str(source), 'source_folder': str(folder),
            'collection': collection, 'teacher_name': teacher, 'max_hands': limit,
            'profitable_only': profitable,
            'source_namespace': namespace, 'group_root': str(group_root)}


def source_files(config):
    """Interleave sessions so a small Pluribus import includes multiple groups."""
    folder = Path(config['source_folder'])
    sessions = defaultdict(list)
    for path in sorted(folder.rglob('*')):
        if path.suffix not in ('.phh', '.phhs') or not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(folder)
        stat = path.stat()
        sessions[str(relative.parent)].append({'path': str(relative), 'size': stat.st_size,
                                              'mtime_ns': stat.st_mtime_ns})
    if not sessions:
        raise ValueError('No .phh or .phhs files were found in this folder')
    files = []
    for group in zip_longest(*sessions.values()):
        files.extend(item for item in group if item is not None)
        if len(files) >= config['max_hands']:
            break
    # A file has at least one entry (a malformed file counts as one skipped entry).
    return files[:config['max_hands']]


def file_hands(folder, entry):
    path = folder / entry['path']
    stat = path.stat()
    if path.is_symlink() or stat.st_size != entry['size'] or stat.st_mtime_ns != entry['mtime_ns']:
        raise ValueError('PHH source changed after import was queued; create a new import')
    if stat.st_size > MAX_SOURCE_BYTES:
        return [None], 'file_over_32_mib', None
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        content = tomllib.loads(raw.decode('utf-8'))
        if path.suffix == '.phhs':
            if not content or not all(isinstance(value, dict) for value in content.values()):
                return [None], 'malformed_phhs', digest
            return list(content.values()), None, digest
        return [content], None, digest
    except (UnicodeError, tomllib.TOMLDecodeError):
        return [None], 'malformed_phh', digest


def iter_hands(config, files, cursor=None, fingerprints=None):
    folder = Path(config['source_folder'])
    cursor = cursor or {'file': 0, 'hand': 0}
    fingerprints = fingerprints if fingerprints is not None else {}
    for file_index in range(cursor['file'], len(files)):
        entry = files[file_index]
        hands, error, digest = file_hands(folder, entry)
        previous = fingerprints.get(entry['path'])
        if previous and digest != previous:
            raise ValueError('A previously read PHH source file changed; create a new import')
        if digest:
            fingerprints[entry['path']] = digest
        start = cursor['hand'] if file_index == cursor['file'] else 0
        relative = (folder / entry['path']).relative_to(Path(config['group_root']))
        session = str(relative.parent)
        if session == '.':
            session = relative.stem if relative.suffix == '.phhs' else 'single-folder'
        group_id = f"phh:{config['source_namespace']}:{session}"
        for hand_index in range(start, len(hands)):
            hand = hands[hand_index]
            canonical = json.dumps(hand, sort_keys=True, separators=(',', ':'), default=str)
            source_id = hashlib.sha256(canonical.encode()).hexdigest()
            next_cursor = {'file': file_index, 'hand': hand_index + 1}
            if hand_index + 1 == len(hands):
                next_cursor = {'file': file_index + 1, 'hand': 0}
            yield hand, error, source_id, group_id, next_cursor


def convert(hand, error, source_id, group_id, config):
    from arena.phh import convert_hand
    if error:
        return {'records': [], 'teachers': [], 'skip_reason': error, 'counters': {}}
    return convert_hand(hand, source_id=source_id, group_id=group_id,
                        teacher_name=config['teacher_name'] or '*',
                        profitable_only=config['profitable_only'],
                        source_namespace=config['source_namespace'])


def preview(body):
    config = settings(body)
    # Preview has its own hard bound even if a much larger import was requested.
    sample_config = dict(config, max_hands=min(PREVIEW_HANDS, config['max_hands']))
    files = source_files(sample_config)
    result = {'sample_hands': 0, 'usable_hands': 0, 'records': 0, 'skipped': {},
              'teachers': [], 'table_sizes': [], 'source_folder': config['source_folder']}
    teachers, sizes = set(), set()
    for hand, error, source_id, group_id, _ in iter_hands(config, files):
        converted = convert(hand, error, source_id, group_id, config)
        result['sample_hands'] += 1
        records = converted['records']
        if records:
            result['usable_hands'] += 1
            result['records'] += len(records)
            teachers.update(record['teacher']['name'] for record in records)
            sizes.update(record['table_size'] for record in records)
        else:
            reason = converted.get('skip_reason') or 'no_eligible_decisions'
            result['skipped'][reason] = result['skipped'].get(reason, 0) + 1
        if result['sample_hands'] >= sample_config['max_hands']:
            break
    result.update(teachers=sorted(teachers), table_sizes=sorted(sizes))
    result['note'] = ('Preview only; no dataset or training was started. Supports 4–6 player '
                      'no-limit Hold’em, 100 BB starting stacks, standard blinds and no antes. '
                      'Hand limit counts source hands inspected, including skipped hands.')
    return result


def queue(server, body):
    from arena.phh import CONVERTER_VERSION
    config = settings(body)
    name = datasets.checked_name(body)
    files = source_files(config)
    # PHH players are dataset teachers, not executable agents in the arena.
    dataset = datasets.new_dataset(server, name, [], config['max_hands'], config,
                                   target_unit='episodes', persist=False)
    dataset.update(kind='phh', processed_hands=0, source='PHH',
                   cursor={'file': 0, 'hand': 0},
                   phh_stats={'processed_hands': 0, 'imported_hands': 0,
                              'skipped': {}, 'counters': {}},
                   engine_config={'engine': 'PHH replay', 'pokerkit_version': '0.7.6',
                                  'converter_version': CONVERTER_VERSION,
                                  'pot_split': 'fractional' if config['source_namespace'] == 'pluribus' else 'source_units',
                                  'stack': 200, 'blinds': [1, 2], 'source_blind_normalization': True})
    folder = server.DATA / 'datasets' / dataset['id']
    datasets.atomic_json(folder / 'sources.json', {'files': files})
    server.save('datasets', dataset)
    return dataset


def execute(server, dataset):
    """Checkpoint batches of complete hands, preserving stable decision IDs."""
    config = dataset['config']
    folder = server.DATA / 'datasets' / dataset['id']
    latest = server.get('datasets', dataset['id'])
    if latest.get('finish_requested') or dataset['processed_hands'] >= config['max_hands']:
        datasets.seal(server, dataset)
        return
    files = json.loads((folder / 'sources.json').read_text())['files']
    fingerprints = dict(dataset.get('source_fingerprints', {}))
    # Metadata validation is inexpensive and catches changes to completed files on resume.
    for entry in files:
        path = Path(config['source_folder']) / entry['path']
        stat = path.stat()
        if path.is_symlink() or stat.st_size != entry['size'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('PHH source changed after import was queued; create a new import')
    pending_records = []
    pending_hands = 0
    batch_processed = 0
    current_size = None
    previous_elapsed = dataset['elapsed_seconds']
    started = time.monotonic()

    def checkpoint():
        nonlocal pending_records, pending_hands, batch_processed
        dataset['elapsed_seconds'] = previous_elapsed + time.monotonic() - started
        dataset['source_fingerprints'] = dict(fingerprints)
        dataset['progress'] = (f"Inspected {dataset['processed_hands']:,} source hands; "
                               f"kept {dataset['phh_stats']['imported_hands']:,} hands")
        if pending_records:
            temporary = folder / 'phh-batch.jsonl'
            try:
                with temporary.open('w', encoding='utf-8') as stream:
                    for record in pending_records:
                        stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                shard = datasets.write_shard(server, dataset, temporary, dataset['games'] + 1, current_size)
                datasets.commit_shard(server, dataset, shard, {'num_hands': pending_hands})
            finally:
                temporary.unlink(missing_ok=True)
        else:
            datasets.save_progress(server, dataset)
        pending_records = []
        pending_hands = 0
        batch_processed = 0

    if datasets.pause_if_requested(server, dataset):
        return
    iterator = iter_hands(config, files, dataset['cursor'], fingerprints)
    while True:
        latest = server.get('datasets', dataset['id'])
        if latest.get('finish_requested') or dataset['processed_hands'] >= config['max_hands']:
            break
        try:
            hand, error, source_id, group_id, next_cursor = next(iterator)
        except StopIteration:
            break
        converted = convert(hand, error, source_id, group_id, config)
        records = converted['records']
        size = records[0]['table_size'] if records else current_size
        if pending_records and size != current_size:
            checkpoint()
        current_size = size
        known_teachers = {teacher['id'] for teacher in dataset['teachers']}
        for record in records:
            teacher = record['teacher']
            if teacher['id'] not in known_teachers:
                dataset['teachers'].append({key: teacher[key] for key in ('id', 'name', 'hash')})
                dataset['teacher_counts'].setdefault(teacher['id'], 0)
                known_teachers.add(teacher['id'])
        dataset['processed_hands'] += 1
        dataset['phh_stats']['processed_hands'] += 1
        dataset['cursor'] = next_cursor
        counters = Counter(dataset['phh_stats']['counters'])
        counters.update(converted.get('counters', {}))
        dataset['phh_stats']['counters'] = dict(counters)
        if records:
            pending_records.extend(records)
            pending_hands += 1
            dataset['phh_stats']['imported_hands'] += 1
        else:
            reason = converted.get('skip_reason') or 'no_eligible_decisions'
            skipped = dataset['phh_stats']['skipped']
            skipped[reason] = skipped.get(reason, 0) + 1
        batch_processed += 1
        if batch_processed >= BATCH_HANDS or server.STOP.is_set() or dataset['id'] in server.PAUSE:
            checkpoint()
            if datasets.pause_if_requested(server, dataset):
                return
    checkpoint()
    if datasets.pause_if_requested(server, dataset):
        return
    datasets.seal(server, dataset)
