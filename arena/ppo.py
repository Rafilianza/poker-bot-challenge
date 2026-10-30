"""Queue PPO experiments while preserving their source BC checkpoint."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

from arena import cloning, ppo_milestones, ppo_snapshots
from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION


def checkpoint_choices(server):
    choices = []
    for job in reversed(server.all_rows('training_jobs')):
        checkpoint = server.DATA / 'training_runs' / job['id'] / 'best.pt'
        eligible = job['status'] == 'completed' and checkpoint.is_file() and checkpoint.with_name('weights.json').is_file()
        reason = '' if eligible else 'Finish BC training before using its checkpoint'
        choices.append({
            'id': job['id'], 'name': job['name'], 'status': job['status'],
            'epoch': job.get('best_epoch', job.get('epoch', 0)), 'epochs': job.get('epochs', 0),
            'eligible': eligible, 'reason': reason, 'records': job.get('records', 0),
            'device': job.get('device'),
        })
    return choices


def state(server):
    jobs = list(reversed(server.all_rows('ppo_jobs')))
    for job in jobs:
        job['folders'] = job_folders(server, job)
        job['snapshot'] = ppo_snapshots.describe(server, job['id'])
        job['milestones'] = ppo_milestones.summaries(server.DATA / 'ppo_runs' / job['id'])
    live_enabled = getattr(server, 'LIVE_PPO_DATA', None) is not None
    live_jobs = []
    live_error = None
    if live_enabled:
        try:
            live_jobs = ppo_snapshots.list_live(server)
        except (OSError, ValueError, sqlite3.Error) as error:
            live_error = str(error)
    return {'checkpoints': checkpoint_choices(server), 'agents': server.all_rows('agents'),
            'ppo_jobs': jobs, 'folder_actions_available': True,
            'placement_rewards_available': True,
            'self_play_available': True,
            'finder_available': sys.platform == 'darwin',
            'development_mode': getattr(server, 'DEVELOPMENT', False),
            'live_source_enabled': live_enabled, 'live_jobs': live_jobs,
            'live_source_error': live_error}


def job_folders(server, job):
    """Resolve only the folders belonging to this registered PPO job."""
    data_folder = server.DATA.resolve()
    folders = {'agent': None, 'checkpoints': None}
    for kind, directory, key in (
        ('checkpoints', 'ppo_runs', job['id']),
        ('agent', 'agents', job.get('agent_id')),
    ):
        if key is None:
            continue
        if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{32}', key):
            raise ValueError('Invalid PPO folder identifier')
        expected_folder = data_folder / directory / key
        folder = expected_folder.resolve()
        if folder != expected_folder:
            raise ValueError('PPO folder must stay inside its storage directory')
        folders[kind] = str(folder)
    return folders


def open_folder(server, key, kind):
    """Open a job's existing agent or checkpoint directory in macOS Finder."""
    if kind not in ('agent', 'checkpoints'):
        raise ValueError('Choose the agent or checkpoints folder')
    job = server.get('ppo_jobs', key)
    folders = job_folders(server, job)
    path = folders[kind]
    if path is None:
        raise ValueError('Create the PPO agent before opening its playable folder')
    if kind == 'agent':
        server.get('agents', job['agent_id'])
    if not Path(path).is_dir():
        raise ValueError('The saved folder could not be found')
    if sys.platform != 'darwin':
        raise ValueError('Open in Finder is available on macOS; use the displayed folder path')
    try:
        # An argument list preserves spaces in paths and never invokes a shell.
        subprocess.run(['/usr/bin/open', path], check=True, capture_output=True,
                       text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('Finder could not open the folder; use the displayed folder path') from exc
    return {'path': path, 'folder': kind}


def integer(body, key, default, minimum, maximum):
    value = body.get(key, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{key} must be an integer from {minimum:,} to {maximum:,}')
    return value


def decimal(body, key, default, minimum, maximum):
    value = body.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{key} must be between {minimum} and {maximum}')
    return float(value)


def queue(server, body):
    name = str(body.get('name', '')).strip()
    if not name or len(name) > 100:
        raise ValueError('Give the PPO student a name of 1–100 characters')
    source = server.get('training_jobs', body.get('source_training_id', ''))
    if source['status'] != 'completed':
        raise ValueError('Choose a completed BC checkpoint; the running BC job stays untouched')
    source_folder = server.DATA / 'training_runs' / source['id']
    source_path = source_folder / 'best.pt'
    if not source_path.is_file():
        raise ValueError('The BC checkpoint is missing')
    weights_path = source_folder / 'weights.json'
    if not weights_path.is_file():
        raise ValueError('The completed BC model metadata is missing')
    exported = json.loads(weights_path.read_text())
    if exported.get('feature_version') != FEATURE_VERSION or exported.get('feature_count') != FEATURE_COUNT:
        raise ValueError('This BC checkpoint has an incompatible observation encoder')
    sizes = body.get('table_sizes', [4, 5, 6])
    if (not isinstance(sizes, list) or not sizes or
            any(type(size) is not int or size not in (4, 5, 6) for size in sizes) or
            len(sizes) != len(set(sizes))):
        raise ValueError('Choose distinct table sizes from 4, 5 and 6')
    training_mode = body.get('training_mode', 'bots')
    if training_mode not in ('bots', 'self_play', 'mixed'):
        raise ValueError('Choose existing bots, self-play or mixed self-play')
    opponent_ids = body.get('opponents', [])
    if (not isinstance(opponent_ids, list) or
            not all(isinstance(key, str) for key in opponent_ids) or
            not max(sizes) - 1 <= len(opponent_ids) <= 60 or
            len(opponent_ids) != len(set(opponent_ids))):
        purpose = 'registered opponents' if training_mode == 'bots' else 'fixed benchmark opponents'
        raise ValueError(f'Choose {max(sizes) - 1}–60 distinct {purpose}')
    opponents = [server.get('agents', key) for key in opponent_ids]
    # Older API clients retain their existing training/evaluation behavior.
    # The updated dashboard explicitly enables the new milestone feature.
    milestone_settings = ppo_milestones.normalize_settings(body.get('milestone_settings', {'enabled': False}))
    if milestone_settings['enabled'] and len(opponents) < 5:
        raise ValueError('Milestone evaluation at 4, 5 and 6 players needs at least five distinct opponents')
    device = body.get('device', 'auto')
    if device not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('Choose Automatic, CPU, Apple GPU or NVIDIA GPU')
    if getattr(server, 'DEVELOPMENT', False):
        if device == 'mps':
            raise ValueError('This development workspace uses CPU to leave the main training GPU available')
        device = 'cpu'
    reward_mode = body.get('reward_mode', 'chips')
    if reward_mode not in ('chips', 'game_placement'):
        raise ValueError('Choose chip profit or game placement as the PPO reward')
    placement_reward = reward_mode == 'game_placement'
    config = {
        'source_training_id': source['id'], 'opponents': opponent_ids,
        'training_mode': training_mode,
        'table_sizes': sorted(sizes), 'device': device,
        'target_episodes': integer(body, 'target_episodes', 10000, 1, 10000000),
        'rollout_episodes': integer(body, 'rollout_episodes', 1000, 1, 10000),
        'update_epochs': integer(body, 'update_epochs', 4, 1, 20),
        'batch_size': integer(body, 'batch_size', 256, 16, 8192),
        'learning_rate': decimal(body, 'learning_rate', 0.0001, 0.000001, 0.01),
        'critic_learning_rate': decimal(body, 'critic_learning_rate', 0.001, 0.000001, 0.01),
        'clip_ratio': decimal(body, 'clip_ratio', 0.2, 0.01, 0.5),
        'entropy_coef': decimal(body, 'entropy_coef', 0.01, 0, 1),
        'bc_coef': decimal(body, 'bc_coef', 0.02, 0, 10),
        'target_kl': decimal(body, 'target_kl', 0.02, 0.001, 0.5),
        'gamma': decimal(body, 'gamma', 1.0, 1, 1),
        'gae_lambda': decimal(body, 'gae_lambda', 1.0 if placement_reward else 0.95, 0, 1),
        'eval_interval': integer(body, 'eval_interval', 5, 1, 1000),
        'eval_hands': integer(body, 'eval_hands', 100 if placement_reward else 10, 1, 100),
        'eval_seeds': integer(body, 'eval_seeds', 3, 1, 20),
        'seed': integer(body, 'seed', 42, 0, 2147483647),
    }
    if training_mode == 'mixed':
        config['self_play_history_limit'] = integer(body, 'self_play_history_limit', 5, 1, 20)
        config['self_play_history_interval'] = integer(body, 'self_play_history_interval', 5, 1, 1000)
    if placement_reward:
        # Placement is measured after a full game. Never train on a partial
        # game or cut its return at a hand boundary.
        config.update(
            reward_mode=reward_mode,
            placement_weight=decimal(body, 'placement_weight', 0.9, 0.5, 1.0),
            chip_scale_bb100=decimal(body, 'chip_scale_bb100', 100.0, 1.0, 10000.0),
        )
        for field in ('target_episodes', 'rollout_episodes'):
            if config[field] % 100:
                raise ValueError(f'{field} must be a multiple of 100 for complete placement-reward games')
        if config['gae_lambda'] != 1.0 or config['eval_hands'] != 100:
            raise ValueError('Placement reward requires GAE lambda 1 and 100-hand evaluation games')
        if milestone_settings['enabled']:
            if milestone_settings['interval'] % 100 or milestone_settings['eval_hands'] != 100:
                raise ValueError('Placement milestones require intervals divisible by 100 and 100-hand games')
    job = {
        'id': uuid.uuid4().hex, 'name': name, 'created': server.now(),
        'status': 'queued', 'phase': 'Queued', 'progress': 'Waiting in the local work queue',
        'config': config, 'source_name': source['name'], 'source_training_id': source['id'],
        'episodes': 0, 'target_episodes': config['target_episodes'], 'decisions': 0,
        'iteration': 0, 'history': [], 'pause_requested': False, 'error': None,
    }
    folder = server.DATA / 'ppo_runs' / job['id']
    folder.mkdir(parents=True)
    try:
        # PPO owns its checkpoint and opponent copies. BC never writes here.
        checkpoint = folder / 'source_bc.pt'
        shutil.copy2(source_path, checkpoint)
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        source_info = {'path': str(checkpoint.resolve()), 'sha256': digest,
                       'training_id': source['id'], 'name': source['name'],
                       'epoch': source.get('best_epoch', source.get('epoch'))}
        frozen_opponents = []
        for opponent in opponents:
            destination = folder / 'opponents' / opponent['id']
            shutil.copytree(server.DATA / 'agents' / opponent['id'], destination)
            frozen_opponents.append({
                'id': opponent['id'], 'name': opponent['name'], 'hash': opponent['hash'],
                'path': str((destination / 'main.py').resolve()),
            })
        request = {'job_id': job['id'], 'feature_version': FEATURE_VERSION,
                   'source_checkpoint': source_info, 'opponents': frozen_opponents, 'config': config}
        cloning.write_json(folder / 'request.json', request)
        ppo_milestones.ensure_settings(folder, request, milestone_settings)
        job['milestone_settings'] = milestone_settings
        job['source_checkpoint'] = source_info
        job['opponents'] = [{key: opponent[key] for key in ('id', 'name', 'hash')} for opponent in opponents]
        server.save('ppo_jobs', job)
    except Exception:
        shutil.rmtree(folder)
        raise
    return job


def control(server, key, action):
    with server.LOCK:
        job = server.get('ppo_jobs', key)
        folder = server.DATA / 'ppo_runs' / key
        if action == 'pause':
            if job['status'] not in ('queued', 'running'):
                raise ValueError('Only queued or running PPO jobs can be paused')
            cloning.write_json(folder / 'control.json', {'action': 'pause'})
            if job['status'] == 'queued':
                job.update(status='paused', phase='Paused', progress='Paused before starting')
            else:
                job.update(pause_requested=True, progress='Pausing; an unfinished iteration will be replayed on resume')
        elif action == 'resume':
            if job['status'] not in ('paused', 'interrupted', 'failed'):
                raise ValueError('Only paused, interrupted or failed PPO jobs can resume')
            cloning.write_json(folder / 'control.json', {'action': 'run'})
            job.update(status='queued', phase='Queued', progress='Waiting to resume the saved PPO checkpoint',
                       pause_requested=False, error=None)
        else:
            raise ValueError('Unknown PPO action')
        server.save('ppo_jobs', job)
        return job


def execute(server, job):
    folder = server.DATA / 'ppo_runs' / job['id']
    with server.LOCK:
        # Re-read controls in case the job was paused after the queue selected it.
        job = server.get('ppo_jobs', job['id'])
        if job['status'] not in ('queued', 'running'):
            return job
    request = json.loads((folder / 'request.json').read_text())
    if not (folder / 'milestone-settings.json').exists():
        # A resumed legacy job keeps its original request bytes and checkpoint
        # hash. Tracking begins from saved progress; old milestones are not invented.
        enabled = len({opponent['id'] for opponent in request['opponents']}) >= 5
        ppo_milestones.ensure_settings(folder, request, {'enabled': enabled})
    if getattr(server, 'DEVELOPMENT', False) and request['config']['device'] != 'cpu':
        raise ValueError('This preview only runs CPU jobs; use the read-only live snapshot controls for main jobs')
    with server.LOCK:
        job = server.get('ppo_jobs', job['id'])
        if job['status'] not in ('queued', 'running'):
            return job
        job.update(status='running', phase='Starting', started=job.get('started', server.now()),
                   error=None, pause_requested=False)
        server.save('ppo_jobs', job)
    status_path = folder / 'status.json'
    status_path.unlink(missing_ok=True)
    command = [sys.executable, '-u', '-m', 'arena.ppo_train_worker', str(folder / 'request.json')]
    stop_started = None
    last_status = None
    with (folder / 'training.log').open('a') as log:
        process = subprocess.Popen(command, cwd=server.ROOT.parent, stdout=log, stderr=log,
                                   start_new_session=os.name != 'nt')
        try:
            while process.poll() is None:
                if server.STOP.is_set() and stop_started is None:
                    cloning.write_json(folder / 'control.json', {'action': 'pause'})
                    stop_started = time.monotonic()
                if stop_started is not None and time.monotonic() - stop_started > 15:
                    server.kill_tree(process)
                    break
                if status_path.exists():
                    status_text = status_path.read_text()
                    if status_text != last_status:
                        update = json.loads(status_text)
                        if update.get('status') in ('paused', 'completed', 'failed'):
                            update = {**update, 'status': 'running', 'phase': 'Finishing'}
                        with server.LOCK:
                            current = server.get('ppo_jobs', job['id'])
                            current.update(update)
                            server.save('ppo_jobs', current)
                        last_status = status_text
                time.sleep(0.25)
        finally:
            server.kill_tree(process)
    with server.LOCK:
        current = server.get('ppo_jobs', job['id'])
        if status_path.exists():
            current.update(json.loads(status_path.read_text()))
        if current['status'] == 'running':
            if server.STOP.is_set():
                current.update(status='interrupted', phase='Interrupted', progress='Resume from the last completed PPO iteration')
            else:
                current.update(status='failed', phase='Failed', error=(folder / 'training.log').read_text()[-2000:] or 'PPO worker stopped unexpectedly')
        current['pause_requested'] = False
        if current['status'] == 'completed':
            current['finished'] = server.now()
        server.save('ppo_jobs', current)
        return current


def export_agent(server, key):
    return cloning.export_agent(server, key, table='ppo_jobs', folder_name='ppo_runs')
