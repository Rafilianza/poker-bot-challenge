"""Loopback-only dashboard, immutable agent snapshots, persistent serial run queue."""
import argparse
import base64
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from arena.scoring import points, table_sizes
from arena.export import lean_public_files
from arena import study, datasets, ppo, grpo, phh_jobs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
HOUSE = ['house:call', 'house:checkfold', 'house:allin', 'house:random']
LIMIT = 20 * 1024 * 1024
LOCK = threading.RLock()
STOP = threading.Event()
CANCEL = set()
PAUSE = set()
DEVELOPMENT = False
LIVE_PPO_DATA = None

def now():
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def connect():
    db = sqlite3.connect(DATA / 'arena.sqlite', timeout=30)
    try:
        with db:
            yield db
    finally:
        db.close()

def init(hold_queued=False):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'datasets').mkdir(exist_ok=True)
    (DATA / 'training_runs').mkdir(exist_ok=True)
    (DATA / 'ppo_runs').mkdir(exist_ok=True)
    (DATA / 'grpo_runs').mkdir(exist_ok=True)
    with connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        for table in ('datasets', 'dataset_shards', 'training_jobs', 'ppo_jobs', 'grpo_jobs'):
            db.execute(f'CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        for table in ('datasets', 'training_jobs', 'ppo_jobs', 'grpo_jobs'):
            for key, payload in db.execute(f'SELECT id,payload FROM {table}').fetchall():
                job = json.loads(payload)
                if job['status'] == 'running' or (hold_queued and job['status'] == 'queued'):
                    job.update(status='interrupted', pause_requested=False, progress='Held on startup; resume to continue saved work')
                    db.execute(f'UPDATE {table} SET payload=? WHERE id=?', (json.dumps(job), key))
        for key, payload in db.execute('SELECT id,payload FROM runs').fetchall():
            run = json.loads(payload)
            if run['status'] == 'running':
                run.update(status='interrupted', error='Server stopped during this run. Queue a new run; partial results are excluded from leaderboard.')
                if run['mode'] == 'study':
                    run.update(error='Server stopped. Resume this study to continue from its saved games.', pause_requested=False)
                if 'training' in run:
                    run['training']['status'] = 'incomplete'
                db.execute('UPDATE runs SET payload=? WHERE id=?', (json.dumps(run), key))
            elif hold_queued and run['status'] == 'queued':
                run.update(status='interrupted', progress='Not started; queue a new run', error='Queued before this launch; queue a new run to start it.')
                if run['mode'] == 'study':
                    run.update(progress='Held on startup; resume when ready', error=None, pause_requested=False)
                if 'training' in run:
                    run['training']['status'] = 'incomplete'
                db.execute('UPDATE runs SET payload=? WHERE id=?', (json.dumps(run), key))

def all_rows(table):
    with connect() as db:
        return [json.loads(row[0]) for row in db.execute(f'SELECT payload FROM {table} ORDER BY rowid')]

def get(table, key):
    with connect() as db:
        row = db.execute(f'SELECT payload FROM {table} WHERE id=?', (key,)).fetchone()
    if not row:
        raise ValueError('Record not found')
    return json.loads(row[0])

def save(table, item):
    # Keep checkpoints and HTTP status changes in the same write order.
    with LOCK, connect() as db:
        db.execute(f'INSERT OR REPLACE INTO {table}(id,payload) VALUES (?,?)', (item['id'], json.dumps(item)))

def register(body):
    name = str(body.get('name', '')).strip()
    if not name or len(name) > 100:
        raise ValueError('Agent name must contain 1–100 characters')
    filename = str(body.get('filename', ''))
    try:
        raw = base64.b64decode(body['content'], validate=True)
    except Exception:
        raise ValueError('Invalid file encoding')
    if len(raw) > LIMIT:
        raise ValueError('Upload exceeds 20 MB')
    files = {}
    if filename.lower().endswith('.py'):
        files['main.py'] = raw
    elif filename.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                total = 0
                for member in archive.infolist():
                    path = PurePosixPath(member.filename)
                    if member.is_dir():
                        continue
                    if path.is_absolute() or '..' in path.parts or '\\' in member.filename or ':' in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Zip contains an unsafe path or symlink')
                    if any(part.endswith(('.', ' ')) or part.split('.')[0].upper() in {'CON','PRN','AUX','NUL', *(f'COM{i}' for i in range(1,10)), *(f'LPT{i}' for i in range(1,10))} for part in path.parts):
                        raise ValueError('Zip contains an unsupported filename')
                    if str(path).casefold() in {p.casefold() for p in files}:
                        raise ValueError('Zip contains duplicate filenames')
                    total += member.file_size
                    if total > LIMIT or len(files) >= 300:
                        raise ValueError('Submission exceeds 20 MB unpacked or 300 files')
                    files[str(path)] = archive.read(member)
        except zipfile.BadZipFile:
            raise ValueError('Invalid zip archive')
    else:
        raise ValueError('Upload a .py file or .zip with main.py at its root')
    if 'main.py' not in files:
        raise ValueError('main.py must be at the zip root')
    digest = hashlib.sha256()
    for path, content in sorted(files.items()):
        digest.update(path.encode() + b'\0' + hashlib.sha256(content).digest())
    agent = {'id': uuid.uuid4().hex, 'name': name, 'hash': digest.hexdigest(), 'created': now(), 'files': len(files), 'bytes': sum(map(len, files.values()))}
    target = DATA / 'agents' / agent['id']
    for path, content in files.items():
        dest = target / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    save('agents', agent)
    return agent

def agent_archive(key):
    agent = get('agents', key)
    folder = DATA / 'agents' / key
    files = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Agent snapshot contains a symlink')
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = path.read_bytes()
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        digest.update(name.encode() + b'\0' + hashlib.sha256(content).digest())
    if digest.hexdigest() != agent['hash']:
        raise ValueError('Registered snapshot has changed; download refused')
    if 'main.py' not in files or len(files) > 300 or sum(map(len, files.values())) > LIMIT:
        raise ValueError('Snapshot exceeds submission packaging limits')
    files = lean_public_files(files)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return stream.getvalue()

def study_preview(body):
    """Validate the study settings and build a deterministic, read-only plan."""
    ids = body.get('agents', [])
    if not isinstance(ids, list) or not all(isinstance(key, str) for key in ids):
        raise ValueError('Choose registered agents')
    if len(ids) != len(set(ids)) or not 4 <= len(ids) <= 80:
        raise ValueError('Choose 4–80 distinct agents for a table-size study')
    agents = [get('agents', key) for key in ids]
    sizes = body.get('table_sizes', [4, 5, 6])
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size not in (4, 5, 6) for size in sizes):
        raise ValueError('Choose table sizes from 4, 5 and 6')
    if len(sizes) != len(set(sizes)) or max(sizes) > len(ids):
        raise ValueError('Choose distinct table sizes with enough selected bots for the largest table')
    seeds = body.get('seeds', ['arena-1', 'arena-2', 'arena-3'])
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 20 or not all(isinstance(seed, str) and 0 < len(seed) <= 100 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Use 1–20 distinct non-empty seeds, up to 100 characters each')
    deals = body.get('deals', 100)
    batches = body.get('batches', 1)
    if type(deals) is not int or not 1 <= deals <= 1000:
        raise ValueError('Hands per game must be an integer from 1 to 1000')
    if type(batches) is not int or not 1 <= batches <= 20:
        raise ValueError('Batches per seed must be an integer from 1 to 20')
    games = len(seeds) * batches * sum((1 if len(ids) == size else len(ids)) * size for size in sizes)
    if games > 25000:
        raise ValueError('This study exceeds 25,000 games. Choose fewer batches, seeds or bots.')
    plan = study.build_plan(sorted(ids), sorted(sizes), sorted(seeds), deals=deals, batches=batches)
    plan.update(tables=plan['total_tables'], games=plan['total_games'], hands=plan['total_hands'])
    names = {agent['id']: agent['name'] for agent in agents}
    for row in plan['coverage']:
        row['name'] = names[row['id']]
    return plan


def queue_run(body):
    mode = body.get('mode')
    if mode not in ('benchmark', 'tournament', 'study'):
        raise ValueError('Unknown run mode')
    plan = study_preview(body) if mode == 'study' else None
    ids = body.get('agents', [])
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Choose distinct agents')
    if not (1 if mode == 'benchmark' else 2) <= len(ids) <= 80:
        raise ValueError('Choose 1–80 benchmark agents or 2–80 tournament agents')
    for key in ids:
        get('agents', key)
    record_training = body.get('record_training', False)
    teachers = body.get('teachers', [])
    if type(record_training) is not bool:
        raise ValueError('Record training data must be true or false')
    if not isinstance(teachers, list) or not all(isinstance(key, str) for key in teachers) or len(teachers) != len(set(teachers)):
        raise ValueError('Choose distinct teacher agents')
    if record_training and not teachers:
        raise ValueError('Select at least one teacher to record training data')
    if any(key not in ids for key in teachers):
        raise ValueError('Teachers must be selected agents in this run')
    if teachers and not record_training:
        raise ValueError('Enable training recording before choosing teachers')
    seeds = body.get('seeds', ['arena-1', 'arena-2', 'arena-3'])
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 20 or not all(isinstance(s, str) and 0 < len(s) <= 100 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Use 1–20 distinct non-empty seeds, up to 100 characters each')
    seeds = sorted(seeds)
    deals = body.get('deals', 100)
    if type(deals) is not int or not 1 <= deals <= 1000:
        raise ValueError('Hands per game must be an integer from 1 to 1000')
    profile_config = {'seeds': seeds, 'deals': deals, 'opponents': HOUSE, 'engine': 'macpoker-0.1.0', 'stack':200, 'blinds':[1,2], 'bank_ms':30000, 'increment_ms':100, 'policy_seed_version':1}
    profile = hashlib.sha256(json.dumps(profile_config, sort_keys=True).encode()).hexdigest()[:16]
    run = {'id': uuid.uuid4().hex, 'mode':mode, 'status':'queued', 'created':now(), 'progress':'Waiting in queue', 'error':None, 'profile':profile, 'profile_config':profile_config, 'config': {'agents':ids, 'seeds':seeds, 'deals':deals}, 'results':[], 'leaderboard':[]}
    run['config'].update(record_training=record_training, teachers=teachers)
    if plan is not None:
        run['config'].update(table_sizes=sorted(body.get('table_sizes', [4, 5, 6])), batches=body.get('batches', 1))
        run['schedule'] = plan.pop('schedule')
        run['plan'] = plan
        run['profile_config'].pop('opponents', None)
        run['profile_config'].update(
            mode='study', table_sizes=run['config']['table_sizes'], batches=run['config']['batches'],
            schedule_version=plan['schedule_version'],
            participants=[{field: get('agents', key)[field] for field in ('id', 'name', 'hash')} for key in sorted(ids)],
        )
        run['profile'] = hashlib.sha256(json.dumps(run['profile_config'], sort_keys=True).encode()).hexdigest()[:16]
    if record_training:
        teacher_versions = []
        for key in teachers:
            agent = get('agents', key)
            teacher_versions.append({field: agent[field] for field in ('id', 'name', 'hash')})
        run['training'] = {
            'schema_version': 1, 'status': 'collecting', 'records': 0,
            'teacher_counts': {key: 0 for key in teachers}, 'excluded_records': 0,
            'files': [], 'teachers': teacher_versions,
        }
    if plan is not None:
        refresh_study(run)
    save('runs',run)
    return run

def kill_tree(proc):
    if proc.poll() is None:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

class Cancelled(Exception):
    pass

def recover_study_game(path, payload):
    """Recover a finished worker result if the server stopped before saving it."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        seats = payload['config']['seats']
        deals = payload['config']['deals']
        if data.get('request_sha256') != expected_hash or data.get('num_hands') != deals:
            return None
        if len(data['hands']) != deals or len(data['chips']) != seats or len(data['verdicts']) != seats:
            return None
        if payload.get('training'):
            training = data['training']
            expected_file = path.with_suffix('.decisions.jsonl')
            if training['file'] != expected_file.name:
                return None
            counts = {teacher['id']: 0 for teacher in payload['training']['teachers']}
            with expected_file.open(encoding='utf-8') as source:
                for line in source:
                    row = json.loads(line)
                    if row['run_id'] != payload['training']['run_id'] or row['game_id'] != payload['training']['game_id']:
                        return None
                    counts[row['teacher']['id']] += 1
            if counts != training['teacher_counts'] or sum(counts.values()) != training['records']:
                return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None

def game(run, agents, seed, offset, number, seed_group=None):
    if STOP.is_set() or run['id'] in CANCEL:
        raise Cancelled()
    folder = DATA / 'runs' / run['id']
    folder.mkdir(parents=True, exist_ok=True)
    work_root = folder / f'work-{number}'
    specs = []
    for agent in agents:
        if agent['id'].startswith('house:'):
            specs.append(agent['id'])
        else:
            # A fresh per-game working copy prevents agents modifying their registered snapshot.
            work = folder / f'work-{number}' / agent['id']
            specs.append(str(work / 'main.py'))
    request = folder / f'game-{number}.request.json'
    result = folder / f'game-{number}.json'
    payload = {'specs':specs, 'config':{'seats':len(agents), 'deals':run['config']['deals'], 'offset':offset, 'seed':seed, 'stack':200, 'sb':1, 'bb':2, 'base_time_ms':30000, 'increment_ms':100}}
    if run['config'].get('record_training', False):
        teachers = []
        for player, agent in enumerate(agents):
            if agent['id'] in run['config']['teachers']:
                teachers.append({'player': player, **{field: agent[field] for field in ('id', 'name', 'hash')}})
        payload['training'] = {
            'run_id': run['id'], 'game_id': number,
            'seed_group': seed_group if seed_group is not None else seed,
            'teachers': teachers,
        }
        if run['mode'] == 'study':
            planned_table = next(table for table in run['schedule'] if table['seed'] == seed)
            payload['training'].update(
                table_size=len(agents), lineup_id=planned_table['table'], batch=planned_table['batch'],
                lineup=[{field: agent[field] for field in ('id', 'name', 'hash')} for agent in agents],
            )
    if run['mode'] == 'study':
        payload['checkpoint'] = True
        data = recover_study_game(result, payload)
        if data is not None:
            if work_root.exists():
                shutil.rmtree(work_root)
            data.pop('hands')
            data['history_file'] = result.name
            return data
        # Interrupted working copies belong only to this uncommitted game.
        if work_root.exists():
            shutil.rmtree(work_root)
    for agent in agents:
        if not agent['id'].startswith('house:'):
            shutil.copytree(DATA / 'agents' / agent['id'], work_root / agent['id'])
    request.write_text(json.dumps(payload), encoding='utf-8')
    with (folder / f'game-{number}.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'game_worker.py'), str(request), str(result)], stdout=log, stderr=log, start_new_session=os.name != 'nt')
        started = time.monotonic()
        try:
            while proc.poll() is None:
                if STOP.is_set() or run['id'] in CANCEL:
                    raise Cancelled()
                if time.monotonic() - started > len(agents) * (30 + run['config']['deals'] * .1 + 4) + 60:
                    raise RuntimeError('Game worker exceeded its wall-clock watchdog. Run excluded from leaderboard.')
                time.sleep(.15)
        finally:
            kill_tree(proc)
            if work_root.exists() and work_root.resolve().parent == folder.resolve():
                shutil.rmtree(work_root)
    if proc.returncode != 0:
        raise RuntimeError((folder / f'game-{number}.log').read_text(encoding='utf-8')[-4000:])
    data = json.loads(result.read_text(encoding='utf-8'))
    data.pop('hands')
    data['history_file'] = result.name
    return data

def play_table(run, agents, seed, round_no, table_no, counter, seed_group=None):
    table = {'round': round_no, 'table':table_no, 'seed':seed, 'agents':[{'id':a['id'],'name':a['name']} for a in agents], 'games':[]}
    run['results'].append(table)
    totals = [0.] * len(agents)
    for offset in range(len(agents)):
        run['progress'] = f'Round {round_no} · table {table_no} · game {offset+1}/{len(agents)} · seed {seed}'
        save('runs', run)
        counter[0] += 1
        result = game(run, agents, seed, offset, counter[0], seed_group=seed_group)
        table['games'].append(result)
        if 'training' in result:
            recorded = result['training']
            training = run['training']
            training['files'].append(recorded['file'])
            training['records'] += recorded['records']
            training['excluded_records'] += recorded['excluded_records']
            for teacher_id, count in recorded['teacher_counts'].items():
                training['teacher_counts'][teacher_id] += count
        for i, p in enumerate(points(result['chips'])):
            totals[i] += p
        save('runs', run)
    table['standings'] = [{'id':a['id'],'game_points':totals[i],'placement_points':points(totals)[i]} for i,a in enumerate(agents)]
    return table

def refresh_study(run):
    """Rebuild totals from completed games, so resume never adds them twice."""
    agents = [get('agents', key) for key in run['config']['agents']]
    run['study'] = study.summarize(run, agents)
    if 'training' not in run:
        return
    training = run['training']
    sizes = [str(size) for size in run['config']['table_sizes']]
    training.update(records=0, excluded_records=0, files=[])
    training['teacher_counts'] = {teacher['id']: 0 for teacher in training['teachers']}
    training['table_size_counts'] = {size: 0 for size in sizes}
    training['teacher_size_counts'] = {teacher['id']: dict.fromkeys(sizes, 0) for teacher in training['teachers']}
    for table in run['results']:
        size = str(table['table_size'])
        for result in table['games']:
            recorded = result.get('training')
            if recorded is None:
                continue
            training['files'].append(recorded['file'])
            training['records'] += recorded['records']
            training['excluded_records'] += recorded['excluded_records']
            training['table_size_counts'][size] += recorded['records']
            for teacher_id, count in recorded['teacher_counts'].items():
                training['teacher_counts'][teacher_id] += count
                training['teacher_size_counts'][teacher_id][size] += count


def pause_run(key):
    run = get('runs', key)
    if run['mode'] != 'study' or run['status'] not in ('queued', 'running'):
        raise ValueError('Only queued or running studies can be paused')
    if run['status'] == 'queued':
        run.update(status='paused', progress='Paused before starting', pause_requested=False)
        if 'training' in run:
            run['training']['status'] = 'incomplete'
    else:
        PAUSE.add(key)
        run['pause_requested'] = True
    save('runs', run)
    return run


def resume_run(key):
    run = get('runs', key)
    if run['mode'] != 'study' or run['status'] not in ('paused', 'interrupted', 'cancelled', 'failed'):
        raise ValueError('Only paused or interrupted studies can be resumed')
    CANCEL.discard(key)
    PAUSE.discard(key)
    run.update(status='queued', progress='Queued to continue from saved games', error=None, pause_requested=False)
    run.pop('finished', None)
    if 'training' in run:
        run['training']['status'] = 'collecting'
        run['training'].pop('download_url', None)
        run['training'].pop('download_urls_by_size', None)
    refresh_study(run)
    save('runs', run)
    return run


def execute_study(run):
    run.update(status='running', started=run.get('started', now()))
    save('runs', run)
    agents = {key: get('agents', key) for key in run['config']['agents']}
    completed_tables = {table['table']: table for table in run['results']}
    game_number = 0
    for planned in run['schedule']:
        table = completed_tables.get(planned['table'])
        if table is None:
            table = {**planned, 'agents': [agents[key] for key in planned['agent_ids']], 'games': []}
            run['results'].append(table)
        for offset in range(planned['table_size']):
            game_number += 1
            if offset < len(table['games']):
                continue
            if STOP.is_set() or run['id'] in CANCEL:
                raise Cancelled()
            if run['id'] in PAUSE:
                PAUSE.discard(run['id'])
                run.update(status='paused', progress='Paused; completed games saved', pause_requested=False)
                if 'training' in run:
                    run['training']['status'] = 'incomplete'
                refresh_study(run)
                save('runs', run)
                return
            run['progress'] = (
                f"Batch {planned['round']}/{run['plan']['batches']} · "
                f"{planned['table_size']}-player table {planned['table']} · "
                f"game {game_number}/{run['plan']['total_games']}"
            )
            save('runs', run)
            result = game(run, table['agents'], planned['seed'], offset, game_number, seed_group=planned['seed_group'])
            table['games'].append(result)
            run['pause_requested'] = run['id'] in PAUSE
            refresh_study(run)
            save('runs', run)
    PAUSE.discard(run['id'])
    refresh_study(run)
    run.update(status='completed', progress='Complete', finished=now(), pause_requested=False)
    if 'training' in run:
        training = run['training']
        training['status'] = 'ready'
        if training['records']:
            training['download_url'] = f"/api/runs/{run['id']}/training/download"
            training['download_urls_by_size'] = {
                size: f"/api/runs/{run['id']}/training/download?table_size={size}"
                for size, count in training['table_size_counts'].items() if count
            }
    save('runs', run)


def execute(run):
    if run['mode'] == 'study':
        return execute_study(run)
    agents = [get('agents', key) for key in run['config']['agents']]
    counter = [0]
    board = {a['id']: {'id':a['id'], 'name':a['name'], 'chips':0, 'hands':0, 'failures':0, 'sets':0, 'wins':0, 'placement_points':0, 'game_points':0, 'playoff_points':0} for a in agents}
    def accumulate(table, placement=True):
        for i,a in enumerate(table['agents']):
            if a['id'] not in board:
                continue
            row = board[a['id']]
            row['chips'] += sum(g['chips'][i] for g in table['games'])
            row['hands'] += sum(g['num_hands'] for g in table['games'])
            row['failures'] += sum(g['verdicts'][i] != 'OK' for g in table['games'])
            row['sets'] += 1
            row['wins'] += sum(g['chips'][i] for g in table['games']) == max(sum(g['chips'][j] for g in table['games']) for j in range(len(table['agents'])))
            if placement:
                row['placement_points'] += table['standings'][i]['placement_points']
                row['game_points'] += table['standings'][i]['game_points']
    for seed in run['config']['seeds']:
        if run['mode'] == 'benchmark':
            for agent in agents:
                table = play_table(run, [agent] + [{'id':s,'name':s} for s in HOUSE], seed, 1, agent['name'], counter, seed_group=seed)
                accumulate(table)
        else:
            order = agents.copy()
            random.Random(seed).shuffle(order)
            round_points = {a['id']:0 for a in agents}
            game_points = {a['id']:0 for a in agents}
            for round_no in range(1,5):
                start = 0
                for table_no, size in enumerate(table_sizes(len(order)), 1):
                    table = play_table(run, order[start:start+size], f'{seed}:round:{round_no}:table:{table_no}', round_no, table_no, counter, seed_group=seed)
                    accumulate(table)
                    for row in table['standings']:
                        round_points[row['id']] += row['placement_points']
                        game_points[row['id']] += row['game_points']
                    start += size
                order.sort(key=lambda a: (-round_points[a['id']], -game_points[a['id']]))
                run['leaderboard'] = list(board.values())
                save('runs', run)
            # Pairwise head-to-head duplicate sets for groups tied across a prize position.
            for value in sorted(set(round_points.values()), reverse=True):
                tied = [a for a in agents if round_points[a['id']] == value]
                above = sum(v > value for v in round_points.values())
                if len(tied) > 1 and above < 3:
                    for i, left in enumerate(tied):
                        for right in tied[i+1:]:
                            table = play_table(run, [left,right], f'{seed}:playoff:{left["hash"]}:{right["hash"]}', 'playoff', f'{left["name"]} vs {right["name"]}', counter, seed_group=seed)
                            accumulate(table, placement=False)
                            for row in table['standings']:
                                board[row['id']]['playoff_points'] += row['game_points']
    for row in board.values():
        row['mbb'] = row['chips'] / 2 / max(1,row['hands']) * 1000
        row['win_rate'] = row['wins'] / max(1,row['sets']) * 100
    run['leaderboard'] = sorted(board.values(), key=(lambda r: -r['mbb']) if run['mode']=='benchmark' else (lambda r: (-r['placement_points'], -r['playoff_points'])))
    for row in run['leaderboard']:
        row['rank'] = 1 + sum((other['mbb'] > row['mbb']) if run['mode']=='benchmark' else ((other['placement_points'], other['playoff_points']) > (row['placement_points'], row['playoff_points'])) for other in run['leaderboard'])
    run.update(status='completed', progress='Complete', finished=now())
    if 'training' in run:
        run['training']['status'] = 'ready'
        if run['training']['records']:
            run['training']['download_url'] = f'/api/runs/{run["id"]}/training/download'
    save('runs', run)

def worker():
    while not STOP.wait(.3):
        with LOCK:
            queued = [r for r in all_rows('runs') if r['status']=='queued']
            work = [('runs', run) for run in queued]
            for table in ('datasets', 'training_jobs', 'ppo_jobs', 'grpo_jobs'):
                work.extend((table, job) for job in all_rows(table) if job['status'] == 'queued')
            if not work:
                continue
            work.sort(key=lambda item: item[1]['created'])
            work_table, run = work[0]
            run.update(status='running', started=run.get('started', now()))
            save(work_table,run)
        try:
            if work_table == 'datasets':
                datasets.execute(sys.modules[__name__], run)
            elif work_table == 'training_jobs':
                from arena import cloning
                cloning.execute(sys.modules[__name__], run)
            elif work_table == 'ppo_jobs':
                ppo.execute(sys.modules[__name__], run)
            elif work_table == 'grpo_jobs':
                grpo.execute(sys.modules[__name__], run)
            else:
                execute(run)
        except Cancelled:
            if work_table != 'runs':
                run = get(work_table, run['id'])
                run.update(status='interrupted', pause_requested=False, progress='Stopped; resume saved work')
                save(work_table, run)
                continue
            run.update(status='cancelled', progress='Cancelled; partial results excluded', finished=now())
            if 'training' in run:
                run['training']['status'] = 'incomplete'
            save('runs',run)
        except Exception as exc:
            if work_table != 'runs':
                run = get(work_table, run['id'])
                run.update(status='failed', error=str(exc), progress='Stopped; completed work is preserved')
                save(work_table, run)
                continue
            run.update(status='failed',error=str(exc),finished=now())
            if 'training' in run:
                run['training']['status'] = 'incomplete'
            save('runs',run)

def state(profile=None):
    runs = all_rows('runs')
    profiles = {}
    for run in runs:
        if run['mode']=='benchmark':
            cfg=run['config']
            profiles[run['profile']]={'id':run['profile'],'label':f'{cfg["deals"]} hands · {len(cfg["seeds"])} seeds · '+', '.join(cfg['seeds'])}
    profile = profile if profile in profiles else next(reversed(profiles), None)
    rows = {}
    for run in sorted(runs, key=lambda r:r.get('finished',r['created'])):
        if run['mode']=='benchmark' and run['profile']==profile and run['status']=='completed':
            for row in run['leaderboard']:
                rows[row['id']]={**row,'run_id':run['id']}
    summaries = [{k:v for k,v in r.items() if k not in ('results', 'schedule')} for r in reversed(runs)]
    lab_active = sum(job['status'] in ('queued', 'running')
                     for table in ('datasets', 'training_jobs', 'ppo_jobs', 'grpo_jobs') for job in all_rows(table))
    return {'agents':all_rows('agents'),'runs':summaries,'leaderboard':sorted(rows.values(),key=lambda r:-r['mbb']),'profiles':list(profiles.values()),'profile':profile,'lab_active':lab_active}

def lab_state():
    from arena import cloning
    recorded = []
    for run in all_rows('runs'):
        if run['status'] == 'completed' and run.get('training', {}).get('records'):
            recorded.append({'id': run['id'], 'created': run['created'], 'mode': run['mode'], 'records': run['training']['records']})
    collection_jobs = [datasets.detail(sys.modules[__name__], job['id']) for job in reversed(all_rows('datasets'))]
    training_jobs = [cloning.detail(sys.modules[__name__], job['id']) for job in reversed(all_rows('training_jobs'))]
    return {'agents': all_rows('agents'), 'datasets': collection_jobs,
            'training_jobs': training_jobs, 'recorded_runs': list(reversed(recorded)),
            'development_mode': DEVELOPMENT,
            'phh': {'available': True, 'default_source_path': str(phh_jobs.DEFAULT_SOURCE),
                    'preview_cpu_only': DEVELOPMENT}}

def training_archive(key, table_size=None):
    """Package a completed run without loading its decision dataset into memory."""
    run = get('runs', key)
    training = run.get('training', {})
    if run['status'] != 'completed' or training.get('status') != 'ready':
        raise ValueError('Training data is available after the recorded run completes')
    if training.get('records', 0) == 0:
        raise ValueError('This run has no valid teacher decisions to download')
    if table_size is not None:
        if run['mode'] != 'study' or type(table_size) is not int or table_size not in run['config']['table_sizes']:
            raise ValueError('Choose a table size included in this study')
        if not training['table_size_counts'].get(str(table_size), 0):
            raise ValueError('No training decisions were recorded at this table size')
    folder = DATA / 'runs' / key
    suffix = '' if table_size is None else f'-{table_size}'
    archive_path = folder / f'training-dataset{suffix}.zip'
    if archive_path.is_file() and not archive_path.is_symlink():
        return archive_path

    # The manifest describes grouping and labels; it contains no spectator cards.
    manifest = {
        'schema_version': 1, 'run_id': key, 'created': run['created'],
        'finished': run['finished'], 'mode': run['mode'],
        'config': run['config'], 'engine_config': dict(run['profile_config']),
        'teachers': training['teachers'], 'records': training['records'],
        'teacher_counts': training['teacher_counts'],
        'excluded_records': training['excluded_records'],
        'participants': [], 'games': [], 'files': [],
        'usage': {
            'features': 'Use only observation fields that your student can observe. Metadata, requested/executed actions, verdicts, seeds and identities are not observation features.',
            'labels': 'executed_action is the engine-normalized action. requested_action preserves the teacher response for audit. Raise amounts are raise-to totals for the current street; call amounts are additional chips.',
            'filter': 'Every decision from a selected teacher with a non-OK final game verdict is excluded, including otherwise successful earlier decisions. Engine fallback actions are never teacher labels.',
            'split': 'Group by seed_group across all files and runs. Keep all rounds, tables and duplicate seat rotations from that group in one data split; do not randomly split decision rows.',
            'privacy': 'Each observation includes only that teacher\'s hole cards, community cards visible before acting, and public history available at the decision. Other teachers\' rows must never supply hidden cards for this row.',
            'history_scope': 'observation.history covers the current hand. Teacher private memory and public observations from earlier hands are not encoded; these records alone cannot exactly clone adaptation across hands.',
            'training': 'This archive contains demonstrations only. It does not train a student model.',
        },
    }
    if run['mode'] in ('tournament', 'study'):
        manifest['engine_config'].pop('opponents', None)
    if run['mode'] == 'study':
        manifest['schedule_version'] = run['plan']['schedule_version']
        manifest['export_filter'] = {'table_size': table_size}
        manifest['table_size_counts'] = dict(training['table_size_counts'])
        manifest['teacher_size_counts'] = dict(training['teacher_size_counts'])
    selected_files = []
    selected_records = 0
    selected_excluded = 0
    selected_counts = {teacher['id']: 0 for teacher in training['teachers']}
    for agent_id in run['config']['agents']:
        agent = get('agents', agent_id)
        manifest['participants'].append({field: agent[field] for field in ('id', 'name', 'hash')})
    for table in run['results']:
        if table_size is not None and table.get('table_size') != table_size:
            continue
        for offset, result in enumerate(table['games']):
            if 'training' in result:
                recorded = result['training']
                entry = {
                    'round': table['round'], 'table': table['table'],
                    'game_seed': table['seed'], 'offset': offset,
                    'players': table['agents'], **recorded,
                }
                if run['mode'] == 'study':
                    entry.update(table_size=table['table_size'], lineup_id=table['table'], batch=table['batch'], seed_group=table['seed_group'])
                manifest['games'].append(entry)
                selected_files.append(recorded['file'])
                selected_records += recorded['records']
                selected_excluded += recorded['excluded_records']
                for teacher_id, count in recorded['teacher_counts'].items():
                    selected_counts[teacher_id] += count
    if table_size is not None:
        size_key = str(table_size)
        manifest.update(records=selected_records, excluded_records=selected_excluded, teacher_counts=selected_counts)
        manifest['table_size_counts'] = {size_key: selected_records}
        manifest['teacher_size_counts'] = {key: {size_key: count} for key, count in selected_counts.items()}

    temporary_path = archive_path.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for filename in selected_files:
                if not re.fullmatch(r'game-\d+\.decisions\.jsonl', filename):
                    raise ValueError('Invalid training data filename')
                path = folder / filename
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Training data file is missing or is a symlink')
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                manifest['files'].append({'name': filename, 'bytes': path.stat().st_size, 'sha256': digest})
                archive.write(path, filename)
            archive.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path

class Handler(BaseHTTPRequestHandler):
    def send_file(self, path, download, content_type='application/zip'):
        # Stream large datasets instead of copying the entire ZIP into RAM.
        with path.open('rb') as stream:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('Content-Disposition', f'attachment; filename="{download}"')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile)

    def send(self,status,data,ctype='application/json',download=None):
        raw = json.dumps(data).encode() if ctype=='application/json' else data
        self.send_response(status)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(raw)

    def allowed(self):
        host=self.headers.get('Host','')
        port=self.server.server_port
        if host not in (f'127.0.0.1:{port}',f'localhost:{port}'):
            return False
        origin=self.headers.get('Origin')
        return origin is None or origin in (f'http://127.0.0.1:{port}', f'http://localhost:{port}')

    def do_GET(self):
        if not self.allowed():
            return self.send(403,{'error':'Local requests only'})
        url=urlsplit(self.path)
        try:
            if url.path=='/api/state':
                return self.send(200,state(parse_qs(url.query).get('profile',[None])[0]))
            if url.path == '/api/research':
                documents = ROOT.parent / 'docs'
                return self.send(200, {
                    'experiments': json.loads((documents / 'experiments.json').read_text(encoding='utf-8')),
                    'sources': json.loads((documents / 'model-sources.json').read_text(encoding='utf-8')),
                })
            if url.path == '/api/lab':
                return self.send(200, lab_state())
            if url.path == '/api/grpo':
                return self.send(200, grpo.state(sys.modules[__name__]))
            match = re.fullmatch(r'/api/grpo-snapshots/([a-f0-9]{64})/download', url.path)
            if match:
                from arena import grpo_snapshots
                format_name = parse_qs(url.query).get('format', ['zip'])[0]
                path = grpo_snapshots.artifact(sys.modules[__name__], match[1], format_name)
                content_type = 'application/zip' if format_name == 'zip' else 'application/octet-stream'
                return self.send_file(path, f'grpo-snapshot-{match[1][:12]}{path.suffix}', content_type)
            match = re.fullmatch(r'/api/grpo/([a-f0-9]{32})', url.path)
            if match:
                return self.send(200, grpo.detail(sys.modules[__name__], match[1]))
            if url.path == '/api/ppo':
                return self.send(200, ppo.state(sys.modules[__name__]))
            match = re.fullmatch(r'/api/ppo-snapshots/([a-f0-9]{32,64})/download', url.path)
            if match:
                from arena import ppo_snapshots
                format_name = parse_qs(url.query).get('format', ['zip'])[0]
                path = ppo_snapshots.artifact(sys.modules[__name__], match[1], format_name)
                content_type = 'application/zip' if format_name == 'zip' else 'application/octet-stream'
                return self.send_file(path, f'ppo-snapshot-{match[1][:12]}{path.suffix}', content_type)
            match = re.fullmatch(r'/api/ppo/([a-f0-9]{32})', url.path)
            if match:
                return self.send(200, get('ppo_jobs', match[1]))
            match = re.fullmatch(r'/api/datasets/([a-f0-9]{32})(/download)?', url.path)
            if match:
                if match[2]:
                    with datasets.ARCHIVE_LOCK:
                        path = datasets.archive(sys.modules[__name__], match[1])
                    return self.send_file(path, f'dataset-{match[1]}.zip')
                return self.send(200, datasets.detail(sys.modules[__name__], match[1]))
            match = re.fullmatch(r'/api/training/([a-f0-9]{32})', url.path)
            if match:
                from arena import cloning
                return self.send(200, cloning.detail(sys.modules[__name__], match[1]))
            match=re.fullmatch(r'/api/agents/([a-f0-9]{32})/download',url.path)
            if match:
                key=match[1]
                return self.send(200,agent_archive(key),'application/zip',f'agent-{key}.zip')
            if re.fullmatch(r'/api/runs/[a-f0-9]{32}',url.path):
                return self.send(200,get('runs',url.path.rsplit('/',1)[-1]))
            match=re.fullmatch(r'/api/runs/([a-f0-9]{32})/training/download',url.path)
            if match:
                size = parse_qs(url.query).get('table_size', [None])[0]
                size = int(size) if size is not None else None
                with LOCK:
                    path = training_archive(match[1], table_size=size)
                suffix = '' if size is None else f'-{size}-players'
                return self.send_file(path, f'training-{match[1]}{suffix}.zip')
            match=re.fullmatch(r'/api/history/([a-f0-9]{32})/(game-\d+\.json)',url.path)
            if match:
                path=DATA/'runs'/match[1]/match[2]
                return self.send(200,json.loads(path.read_text(encoding='utf-8')))
            mapping={'/':('index.html','text/html; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/lab.js':('lab.js','text/javascript; charset=utf-8'),'/ppo.js':('ppo.js','text/javascript; charset=utf-8'),'/grpo.js':('grpo.js','text/javascript; charset=utf-8')}
            if url.path in mapping:
                name,mime=mapping[url.path]
                return self.send(200,(ROOT/'static'/name).read_bytes(),mime)
            if url.path == '/workspace.js':
                return self.send(200, (ROOT / 'static/workspace.js').read_bytes(), 'text/javascript; charset=utf-8')
            document = url.path.removeprefix('/docs/')
            if url.path.startswith('/docs/') and document in (
                'PIPELINE.md', 'EXPERIMENTS.md', 'CREDITS.md', 'SETUP.md',
            ):
                return self.send(200, (ROOT.parent / 'docs' / document).read_bytes(), 'text/plain; charset=utf-8')
            self.send(404,{'error':'Not found'})
        except (ValueError,FileNotFoundError) as exc:
            self.send(404,{'error':str(exc)})

    def do_POST(self):
        if not self.allowed() or self.headers.get('Content-Type','').split(';')[0]!='application/json':
            return self.send(403,{'error':'Use same-origin JSON requests'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 29*1024*1024:
                raise ValueError('Request too large or empty')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):
                raise ValueError('Expected JSON object')
            match = re.fullmatch(r'/api/grpo/([a-f0-9]{32})/snapshot', self.path)
            if match:
                # Snapshot validation never holds the shared training queue lock.
                from arena import grpo_snapshots
                result = grpo_snapshots.prepare(sys.modules[__name__], match[1],
                                                kind=body.get('kind', 'best'),
                                                milestone_id=body.get('milestone_id'))
                download = f"/api/grpo-snapshots/{result['snapshot_id']}/download"
                result['zip_url'] = download + '?format=zip' if result.get('zip_available') else None
                result['checkpoint_url'] = download + '?format=pt' if result.get('checkpoint_available') else None
                return self.send(200, result)
            match = re.fullmatch(r'/api/grpo/([a-f0-9]{32})/open-folder', self.path)
            if match:
                return self.send(200, grpo.open_folder(sys.modules[__name__], match[1], body.get('folder')))
            match = re.fullmatch(r'/api/ppo/(live/)?([a-f0-9]{32})/snapshot', self.path)
            if match:
                # CPU export validation runs outside the shared work-queue lock.
                from arena import ppo_snapshots
                result = ppo_snapshots.prepare(sys.modules[__name__], match[2],
                                               kind=body.get('kind', 'best'), live=bool(match[1]),
                                               milestone_id=body.get('milestone_id'))
                download = f"/api/ppo-snapshots/{result['snapshot_id']}/download"
                result['zip_url'] = download + '?format=zip' if result.get('zip_available') else None
                result['checkpoint_url'] = download + '?format=checkpoint' if result.get('checkpoint_available') else None
                return self.send(200, result)
            if self.path == '/api/phh/preview':
                return self.send(200, phh_jobs.preview(body))
            if self.path == '/api/phh/import':
                return self.send(201, phh_jobs.queue(sys.modules[__name__], body))
            match = re.fullmatch(r'/api/ppo/([a-f0-9]{32})/open-folder', self.path)
            if match:
                return self.send(200, ppo.open_folder(sys.modules[__name__], match[1], body.get('folder')))
            with LOCK:
                if self.path == '/api/grpo':
                    return self.send(201, grpo.queue(sys.modules[__name__], body))
                match = re.fullmatch(r'/api/grpo/([a-f0-9]{32})/(pause|resume|export)', self.path)
                if match:
                    if match[2] == 'export':
                        return self.send(200, grpo.export_agent(sys.modules[__name__], match[1]))
                    return self.send(200, grpo.control(sys.modules[__name__], match[1], match[2]))
                if self.path == '/api/ppo':
                    return self.send(201, ppo.queue(sys.modules[__name__], body))
                match = re.fullmatch(r'/api/ppo/([a-f0-9]{32})/(pause|resume|export)', self.path)
                if match:
                    if match[2] == 'export':
                        return self.send(200, ppo.export_agent(sys.modules[__name__], match[1]))
                    return self.send(200, ppo.control(sys.modules[__name__], match[1], match[2]))
                if self.path == '/api/datasets':
                    return self.send(201, datasets.queue(sys.modules[__name__], body))
                if self.path == '/api/datasets/import':
                    return self.send(201, datasets.import_run(sys.modules[__name__], body))
                match = re.fullmatch(r'/api/datasets/([a-f0-9]{32})/(pause|resume|finish)', self.path)
                if match:
                    return self.send(200, datasets.control(sys.modules[__name__], match[1], match[2], body))
                if self.path == '/api/training':
                    from arena import cloning
                    return self.send(201, cloning.queue(sys.modules[__name__], body))
                match = re.fullmatch(r'/api/training/([a-f0-9]{32})/(pause|resume|finish|export)', self.path)
                if match:
                    from arena import cloning
                    if match[2] == 'export':
                        return self.send(200, cloning.export_agent(sys.modules[__name__], match[1]))
                    return self.send(200, cloning.control(sys.modules[__name__], match[1], match[2]))
                if self.path=='/api/agents':
                    return self.send(201,register(body))
                if self.path=='/api/runs':
                    return self.send(201,queue_run(body))
                if self.path=='/api/study/preview':
                    plan = study_preview(body)
                    return self.send(200,{key: value for key, value in plan.items() if key != 'schedule'})
                match=re.fullmatch(r'/api/runs/([a-f0-9]{32})/(pause|resume)',self.path)
                if match:
                    action = pause_run if match[2] == 'pause' else resume_run
                    return self.send(200,action(match[1]))
                match=re.fullmatch(r'/api/runs/([a-f0-9]{32})/cancel',self.path)
                if match:
                    run=get('runs',match[1])
                    if run['status']=='queued':
                        run['status']='cancelled'
                        if 'training' in run:
                            run['training']['status'] = 'incomplete'
                        save('runs',run)
                    elif run['status']=='running':
                        CANCEL.add(run['id'])
                    return self.send(200,{'ok':True})
            self.send(404,{'error':'Not found'})
        except (ValueError,KeyError,TypeError) as exc:
            self.send(400,{'error':str(exc)})

def main():
    global DATA, DEVELOPMENT, LIVE_PPO_DATA
    parser = argparse.ArgumentParser(description='Run the local Poker Arena dashboard.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='Folder for registered agents and experiment history')
    parser.add_argument('--open-browser', action='store_true', help='Open the dashboard in your default browser')
    parser.add_argument('--hold-queued', action='store_true', help='Keep previously queued runs from starting automatically; queue new runs in the dashboard')
    parser.add_argument('--development', action='store_true', help='Show the separate-workspace notice and restrict PPO and GRPO to CPU')
    parser.add_argument('--live-ppo-data', type=Path, help='Read-only PPO source for snapshot downloads; never starts or controls its jobs')
    args = parser.parse_args()
    DATA = args.data_dir.expanduser().resolve()
    DEVELOPMENT = args.development
    LIVE_PPO_DATA = args.live_ppo_data.expanduser().resolve() if args.live_ppo_data else None
    if LIVE_PPO_DATA is not None:
        if LIVE_PPO_DATA == DATA or LIVE_PPO_DATA.is_relative_to(DATA) or DATA.is_relative_to(LIVE_PPO_DATA):
            parser.error('Live PPO data and this dashboard data must be separate directories')
        if not (LIVE_PPO_DATA / 'arena.sqlite').is_file():
            parser.error('The read-only live PPO source has no arena.sqlite database')

    # Bind first so an already-running dashboard cannot affect stored runs.
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.exit(1, f'Port {args.port} is already in use. Open http://127.0.0.1:{args.port} if Poker Arena is already running, or choose another --port.\n')
        parser.exit(1, f'Could not start Poker Arena: {exc}\n')

    with server:
        init(hold_queued=args.hold_queued)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Poker Arena: {url}', flush=True)
        print(f'Agent files and results: {DATA}', flush=True)
        try:
            if args.open_browser:
                try:
                    webbrowser.open(url)
                except webbrowser.Error:
                    print(f'Open {url} in your browser.', flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            STOP.set()
            thread.join(timeout=15)

if __name__=='__main__':
    main()
