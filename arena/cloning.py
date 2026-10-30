"""Persistent behaviour-cloning jobs and scaffold-compatible student exports."""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import zipfile

from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION
from arena.ppo_context import CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT

EXPORT_LOCK = threading.Lock()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


def queue(server, body):
    name = str(body.get('name', '')).strip()
    if not name or len(name) > 100:
        raise ValueError('Student name must contain 1–100 characters')
    dataset_ids = body.get('dataset_ids', [])
    if (not isinstance(dataset_ids, list) or not dataset_ids or
            not all(isinstance(key, str) for key in dataset_ids) or len(dataset_ids) != len(set(dataset_ids))):
        raise ValueError('Choose one or more distinct saved datasets')
    datasets = [server.get('datasets', key) for key in dataset_ids]
    if any(dataset['status'] != 'ready' for dataset in datasets):
        raise ValueError('Only completed, ready datasets can be used for training')
    teachers = {}
    manifests = []
    for dataset in datasets:
        folder = server.DATA / 'datasets' / dataset['id']
        manifest_path = folder / 'manifest.json'
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not manifest.get('files') or not manifest.get('records'):
            raise ValueError('A selected dataset contains no demonstrations')
        for teacher in manifest['teachers']:
            if teacher['id'] in teachers and teachers[teacher['id']].get('hash') != teacher.get('hash'):
                raise ValueError('The selected datasets disagree about a teacher version')
            teachers[teacher['id']] = teacher
        manifests.append({'id': dataset['id'], 'path': str(folder.resolve()),
                          'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()})
    selected_teachers = body.get('teachers', sorted(teachers))
    if (not isinstance(selected_teachers, list) or not selected_teachers or
            not all(isinstance(key, str) and key in teachers for key in selected_teachers) or
            len(selected_teachers) != len(set(selected_teachers))):
        raise ValueError('Choose distinct teachers present in the selected datasets')
    sizes = body.get('table_sizes', [4, 5, 6])
    if (not isinstance(sizes, list) or not sizes or
            any(type(size) is not int or size not in (4, 5, 6) for size in sizes) or len(sizes) != len(set(sizes))):
        raise ValueError('Choose table sizes from 4, 5 and 6')
    epochs = body.get('epochs', 10)
    batch_size = body.get('batch_size', 256)
    learning_rate = body.get('learning_rate', 0.001)
    if type(epochs) is not int or not 1 <= epochs <= 200:
        raise ValueError('Epochs must be an integer from 1 to 200')
    if type(batch_size) is not int or not 16 <= batch_size <= 8192:
        raise ValueError('Batch size must be an integer from 16 to 8192')
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or not 0.000001 <= learning_rate <= 0.1):
        raise ValueError('Learning rate must be between 0.000001 and 0.1')
    balanced = body.get('balanced', True)
    if type(balanced) is not bool:
        raise ValueError('Balanced teacher sampling must be true or false')
    teacher_weights = body.get('teacher_weights', {})
    if not isinstance(teacher_weights, dict):
        raise ValueError('Teacher weights must map teacher IDs to positive numbers')
    for teacher, weight in teacher_weights.items():
        if (teacher not in selected_teachers or type(weight) not in (int, float)
                or not math.isfinite(weight) or not 0 < weight <= 1000):
            raise ValueError('Teacher weights must be positive numbers up to 1000 for selected teachers')
    device = body.get('device', 'auto')
    if device not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('Choose Auto, CPU, Apple GPU or NVIDIA GPU for training')
    if getattr(server, 'DEVELOPMENT', False):
        if device == 'mps':
            raise ValueError('This preview uses CPU only to leave the main training GPU available')
        device = 'cpu'
    config = {'dataset_ids': dataset_ids, 'teachers': sorted(selected_teachers),
              'table_sizes': sorted(sizes), 'epochs': epochs, 'batch_size': batch_size,
              'learning_rate': float(learning_rate), 'balanced': balanced,
              'teacher_weights': teacher_weights, 'device': device}
    if getattr(server, 'DEVELOPMENT', False):
        config['cpu_threads'] = 1
    job = {'id': uuid.uuid4().hex, 'name': name, 'created': server.now(),
           'status': 'queued', 'config': config, 'phase': 'Queued', 'progress': 'Waiting to train',
           'teachers': [teachers[key] for key in sorted(selected_teachers)],
           'dataset_names': {dataset['id']: dataset['name'] for dataset in datasets},
           'epoch': 0, 'epochs': epochs, 'history': [], 'finish_requested': False}
    request = {'job_id': job['id'], 'config': config, 'datasets': manifests,
               'feature_version': FEATURE_VERSION}
    write_json(server.DATA / 'training_runs' / job['id'] / 'request.json', request)
    server.save('training_jobs', job)
    return job


def control(server, key, action):
    with server.LOCK:
        job = server.get('training_jobs', key)
        folder = server.DATA / 'training_runs' / key
        if action == 'pause':
            if job['status'] not in ('queued', 'running'):
                raise ValueError('Only queued or running training jobs can be paused')
            write_json(folder / 'control.json', {'action': 'pause'})
            if job['status'] == 'queued':
                job.update(status='paused', phase='Paused', progress='Paused before training started')
            else:
                job['pause_requested'] = True
                job['progress'] = 'Pausing at the next batch boundary'
        elif action == 'resume':
            if job['status'] not in ('paused', 'interrupted', 'failed'):
                raise ValueError('Only paused, interrupted or failed training jobs can be resumed')
            finishing = job.get('finish_requested', False)
            write_json(folder / 'control.json', {'action': 'finish' if finishing else 'run'})
            job.update(status='queued', phase='Queued', progress='Waiting to resume',
                       pause_requested=False, error=None)
            if finishing:
                job['progress'] = 'Waiting to finish using the best saved checkpoint'
        elif action == 'finish':
            # A repeated click after completion is harmless. The selected model
            # has already been evaluated and exported to weights.json.
            if job['status'] == 'completed':
                return detail(server, key)
            if server.STOP.is_set() or (job['status'] == 'running' and job.get('pause_requested')):
                raise ValueError('Wait until training is paused, then finish using its best checkpoint')
            if not can_finish(server, job):
                raise ValueError('Finish is available after at least one validated epoch has been saved')
            write_json(folder / 'control.json', {'action': 'finish'})
            job.update(finish_requested=True, pause_requested=False, error=None)
            if job['status'] == 'running':
                job['progress'] = 'Finishing from the best saved checkpoint; unfinished epoch updates will be discarded'
            else:
                job.update(status='queued', phase='Queued',
                           progress='Waiting to finish using the best saved checkpoint')
        else:
            raise ValueError('Unknown training control')
        server.save('training_jobs', job)
        return job


def can_finish(server, job):
    eligible = job['status'] in ('queued', 'running', 'paused', 'interrupted', 'failed')
    pausing = job['status'] == 'running' and job.get('pause_requested', False)
    checkpoint = server.DATA / 'training_runs' / job['id'] / 'best.pt'
    return eligible and not pausing and not server.STOP.is_set() and checkpoint.is_file()


def detail(server, key):
    job = server.get('training_jobs', key)
    return {**job, 'can_finish': can_finish(server, job)}


def execute(server, job):
    folder = server.DATA / 'training_runs' / job['id']
    if getattr(server, 'DEVELOPMENT', False):
        request = json.loads((folder / 'request.json').read_text())
        config = request.get('config', {})
        # Preview-created jobs already carry an explicit CPU thread budget.
        # Reapply that limit on resume, but leave unmarked legacy GPU requests
        # unchanged instead of silently moving them into this preview.
        if config.get('device') != 'cpu' and 'cpu_threads' not in config:
            raise ValueError('This preview can only execute CPU BC jobs; create a new CPU job')
    with server.LOCK:
        # A control request may arrive after the queue selected this job. Read
        # its current flags instead of overwriting them with the stale copy.
        job = server.get('training_jobs', job['id'])
        if job['status'] not in ('queued', 'running'):
            return job
        # Resumed jobs must respect the development workspace's CPU limit too.
        if getattr(server, 'DEVELOPMENT', False):
            request_path = folder / 'request.json'
            request = json.loads(request_path.read_text())
            request['config'].update(device='cpu', cpu_threads=1)
            write_json(request_path, request)
            job['config'].update(device='cpu', cpu_threads=1)
        job.update(status='running', phase='Starting', started=job.get('started', server.now()),
                   error=None)
        server.save('training_jobs', job)
    status_path = folder / 'status.json'
    status_path.unlink(missing_ok=True)
    command = [sys.executable, '-u', '-m', 'arena.bc_train_worker', str(folder / 'request.json')]
    stop_started = None
    last_status = None
    with (folder / 'training.log').open('a', encoding='utf-8') as log:
        environment = None
        if getattr(server, 'DEVELOPMENT', False):
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1')
        process = subprocess.Popen(command, cwd=server.ROOT.parent, stdout=log, stderr=log,
                                   env=environment,
                                   start_new_session=os.name != 'nt')
        try:
            while process.poll() is None:
                if server.STOP.is_set() and stop_started is None:
                    write_json(folder / 'control.json', {'action': 'pause'})
                    stop_started = time.monotonic()
                if stop_started is not None and time.monotonic() - stop_started > 15:
                    server.kill_tree(process)
                    break
                if status_path.exists():
                    status_text = status_path.read_text()
                    if status_text != last_status:
                        update = json.loads(status_text)
                        # Publish terminal states only after the process has exited.
                        # Otherwise Resume can race with this worker's final save.
                        if update.get('status') in ('paused', 'completed', 'failed'):
                            update = {**update, 'status': 'running', 'phase': 'Finishing'}
                        with server.LOCK:
                            current = server.get('training_jobs', job['id'])
                            current.update(update)
                            server.save('training_jobs', current)
                        last_status = status_text
                time.sleep(0.25)
        finally:
            server.kill_tree(process)
    with server.LOCK:
        current = server.get('training_jobs', job['id'])
        if status_path.exists():
            current.update(json.loads(status_path.read_text()))
        if current['status'] == 'running':
            if server.STOP.is_set():
                current.update(status='interrupted', phase='Interrupted',
                               progress='Server stopped. Resume from the last completed epoch.')
            else:
                log_tail = (folder / 'training.log').read_text()[-2000:]
                current.update(status='failed', phase='Failed', error=log_tail or 'Training process stopped unexpectedly')
        current['pause_requested'] = False
        if current['status'] == 'completed':
            current['finished'] = server.now()
        server.save('training_jobs', current)
    return current


def playable_files(name, weights_bytes, context=None):
    """Build the shared SDK bot and encoder used by every student export."""
    model = json.loads(weights_bytes) if weights_bytes else {}
    input_count = model.get('feature_count', FEATURE_COUNT)
    context_version = model.get('context_version')
    if 'feature_version' in model and (type(model['feature_version']) is not int or model['feature_version'] != FEATURE_VERSION):
        raise ValueError('This student has an incompatible feature version')
    if type(input_count) is int and input_count == PLACEMENT_FEATURE_COUNT and type(context_version) is int and context_version == CONTEXT_VERSION:
        model_context = True
    elif type(input_count) is int and input_count == FEATURE_COUNT and context_version is None:
        model_context = False
    else:
        raise ValueError('This student has incompatible feature or context metadata')
    if context is None:
        context = model_context
    elif weights_bytes and context != model_context:
        raise ValueError('The export context does not match the saved model')
    extra_import = '\nfrom ppo_context import GameContext' if context else ''
    context_hooks = '''
    def __init__(self):
        self.context = GameContext()

    def on_match_start(self, info):
        self.context.send(info)

    def on_hand_end(self, info):
        self.context.send(info)
''' if context else ''
    source = '''"""A trained poker student. Inference needs only Python and macpoker."""
import json
from pathlib import Path
from macpoker import Bot
from bc_features import choose_action, encode_observation, forward_logitsEXTRA_IMPORT

MODEL = json.loads(Path(__file__).with_name('weights.json').read_text())


class MyBot(Bot):
    name = STUDENT_NAME
CONTEXT_HOOKS

    def act(self, state):
        observation = state.raw()
        features = FEATURE_ENCODING
        logits = forward_logits(features, MODEL['layers'])
        action = choose_action(observation, logits)
        if action['action'] == 'raise':
            return state.raise_to(action['amount'])
        if action['action'] == 'fold':
            return state.fold()
        return state.call() if state.to_call else state.check()
'''
    source = source.replace('EXTRA_IMPORT', extra_import)
    source = source.replace('CONTEXT_HOOKS', context_hooks)
    source = source.replace('FEATURE_ENCODING', 'self.context.encode(observation)' if context else 'encode_observation(observation)')
    source = source.replace('STUDENT_NAME', repr(name))
    files = {'main.py': source.encode(), 'weights.json': weights_bytes,
             'bc_features.py': Path(__file__).with_name('bc_features.py').read_bytes()}
    if context:
        files['ppo_context.py'] = Path(__file__).with_name('ppo_context.py').read_bytes()
    return files


def export_agent(server, key, table='training_jobs', folder_name='training_runs'):
    """Verify the standard-library student at all table sizes, then register it."""
    with EXPORT_LOCK:
        job = server.get(table, key)
        if job.get('agent_id'):
            return server.get('agents', job['agent_id'])
        if job['status'] != 'completed':
            raise ValueError('Complete training before creating the student agent')
        folder = server.DATA / folder_name / key
        model = json.loads((folder / 'weights.json').read_text())
        if model.get('feature_version') != FEATURE_VERSION:
            raise ValueError('This student uses a different encoder version. Retrain before exporting.')
        export_folder = folder / 'student'
        export_folder.mkdir(exist_ok=True)
        files = playable_files(job['name'], (folder / 'weights.json').read_bytes())
        for name, content in files.items():
            (export_folder / name).write_bytes(content)
        validation = []
        for size in (2, 3, 4, 5, 6):
            request_path = folder / f'export-check-{size}.request.json'
            result_path = folder / f'export-check-{size}.json'
            request = {'specs': [str(export_folder / 'main.py')] + ['house:call'] * (size - 1),
                       'config': {'seats': size, 'deals': 3, 'seed': 'student-export-' + key,
                                  'base_time_ms': 30000, 'increment_ms': 100}}
            write_json(request_path, request)
            process = subprocess.Popen([sys.executable, str(server.ROOT / 'game_worker.py'),
                                        str(request_path), str(result_path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=os.name != 'nt')
            try:
                _, errors = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill_tree(process)
                raise ValueError('Student validation timed out; the agent was not registered')
            if process.returncode != 0 or not result_path.exists():
                raise ValueError('Student validation failed: ' + errors.decode(errors='replace')[-1500:])
            result = json.loads(result_path.read_text())
            if any(verdict != 'OK' for verdict in result['verdicts']):
                raise ValueError('Student produced a runtime error during validation: ' + str(result['verdicts']))
            validation.append({'table_size': size, 'hands': result['num_hands'], 'verdicts': result['verdicts']})
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        agent = server.register({'name': job['name'], 'filename': 'student.zip',
                                 'content': base64.b64encode(stream.getvalue()).decode()})
        job.update(agent_id=agent['id'], export_validation=validation, exported=server.now())
        server.save(table, job)
        return agent
