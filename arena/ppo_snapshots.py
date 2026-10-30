"""Read saved policies without changing a training job or registering an agent."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone

from arena import cloning
from arena.bc_features import FEATURE_VERSION

EXPORT_VERSION = 1
EXPORT_LOCK = threading.Lock()


def safe_path(root, *parts):
    """Reject symlinks and path traversal within our known storage directories."""
    root = Path(root).resolve()
    expected = root.joinpath(*parts)
    resolved = expected.resolve()
    if resolved != expected or not resolved.is_relative_to(root):
        raise ValueError('Snapshot path must stay inside its storage directory')
    return resolved


def live_rows(server):
    configured = getattr(server, 'LIVE_PPO_DATA', None)
    if configured is None:
        return []
    database = safe_path(configured, 'arena.sqlite')
    if not database.is_file():
        return []
    # Never import or initialize the source server. This connection cannot write.
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        connection.execute('PRAGMA query_only = ON')
        rows = connection.execute('SELECT payload FROM ppo_jobs ORDER BY rowid DESC').fetchall()
        return [json.loads(row[0]) for row in rows]
    except sqlite3.DatabaseError as error:
        raise ValueError('The read-only PPO source database could not be read') from error
    finally:
        connection.close()


def source_job(server, key, live=False):
    if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{32}', key):
        raise ValueError('Invalid PPO job identifier')
    if live:
        data = getattr(server, 'LIVE_PPO_DATA', None)
        if data is None:
            raise ValueError('Read-only live PPO browsing is not enabled')
        job = next((job for job in live_rows(server) if job.get('id') == key), None)
        if job is None:
            raise ValueError('Live PPO job not found')
    else:
        data = server.DATA
        job = server.get('ppo_jobs', key)
    return job, safe_path(data, 'ppo_runs', key)


def availability(folder):
    checkpoint = safe_path(folder, 'best.pt')
    available = checkpoint.is_file()
    details = {'kind': 'best', 'available': available, 'selection_metric': None,
               'metadata_source': 'verified_checkpoint_on_export',
               'reason': '' if available else 'No saved training-best checkpoint yet'}
    if available:
        information = checkpoint.stat()
        details.update(size_bytes=information.st_size,
                       modified_at=datetime.fromtimestamp(information.st_mtime, timezone.utc).isoformat())
    return details


def describe(server, key, live=False):
    """Cheap polling metadata; exact saved progress is read only during export."""
    job, folder = source_job(server, key, live)
    from arena import ppo_milestones
    return {'job_id': key, 'name': job['name'], 'status': job['status'],
            'source': 'live' if live else 'local', 'read_only': True,
            'best': availability(folder), 'milestones': ppo_milestones.summaries(folder)}


def list_live(server):
    result = []
    data = getattr(server, 'LIVE_PPO_DATA', None)
    for job in live_rows(server):
        key = job.get('id')
        if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{32}', key):
            continue
        folder = safe_path(data, 'ppo_runs', key)
        best = availability(folder)
        # These are the job's reported counters, not the saved best actor's age.
        summary = {
            'id': key, 'name': job.get('name', key), 'status': job.get('status', 'unknown'),
            'episodes': job.get('episodes', 0), 'iteration': job.get('iteration', 0),
            'target_episodes': job.get('target_episodes', 0),
            'source': 'live', 'read_only': True, 'best_available': best['available'],
            'snapshot': {'job_id': key, 'source': 'live', 'read_only': True,
                         'best': best, 'milestones': []},
        }
        # Show recorded learning curves without copying weights or making a local job.
        for field in (
            'created', 'source_name', 'config', 'history', 'training_mode', 'reward_mode',
            'selection_metric', 'baseline', 'baseline_test', 'validation',
            'latest_validation', 'best_validation', 'best_score', 'best_iteration',
            'test', 'device', 'decisions', 'elapsed_seconds',
        ):
            if field in job:
                summary[field] = job[field]
        result.append(summary)
    return result


def stable_bytes(path):
    """One descriptor keeps an atomic best.pt replacement from mixing versions."""
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        with os.fdopen(os.open(path, flags), 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 512 * 1024 * 1024:
                raise ValueError('Snapshot checkpoint must be a regular file under 512 MB')
            data = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError('The saved checkpoint is missing or cannot be read') from error
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('The checkpoint changed while reading; retry the export')
    if len(data) != before.st_size:
        raise ValueError('The saved checkpoint is incomplete')
    return data


def run_worker(folder):
    # CPU limits apply only to this separate process, never the training server.
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-m', 'arena.ppo_snapshot_worker', str(folder)]
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent.parent,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=environment, start_new_session=os.name != 'nt')
    try:
        _, errors = process.communicate(timeout=90)
    except subprocess.TimeoutExpired as error:
        if os.name == 'nt':
            # The worker starts SDK games, which start their own bot processes.
            # Killing only the worker can leave those children holding pipes open.
            try:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.stdout.close()
            process.stderr.close()
        raise ValueError('Snapshot validation timed out; training was not changed') from error
    if process.returncode:
        if os.name != 'nt':
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise ValueError('Snapshot validation failed: ' + errors.decode(errors='replace')[-1800:])


def prepare(server, key, kind='best', live=False, milestone_id=None):
    """Copy exact saved bytes, verify on CPU, and publish an isolated export."""
    if kind not in ('best', 'milestone'):
        raise ValueError('Choose a training-best or milestone snapshot')
    job, folder = source_job(server, key, live)
    if live and Path(server.DATA).resolve() == Path(server.LIVE_PPO_DATA).resolve():
        raise ValueError('Live exports require a separate preview data directory')
    source_metadata = None
    checkpoint = safe_path(folder, 'best.pt')
    if kind == 'milestone':
        from arena import ppo_milestones
        source_metadata = ppo_milestones.snapshot(folder, milestone_id)
        checkpoint = Path(source_metadata['checkpoint_path'])
    elif milestone_id is not None:
        raise ValueError('A milestone identifier only applies to a milestone export')

    checkpoint_bytes = stable_bytes(checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    if source_metadata and source_metadata.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('The milestone checkpoint checksum changed')
    context = job.get('config', {}).get('reward_mode', 'chips') == 'game_placement'
    if source_metadata is not None:
        context = source_metadata.get('context_version') is not None
    files = cloning.playable_files(job['name'], b'', context=context)
    identity = {'export_version': EXPORT_VERSION, 'job_id': key,
                'source': 'live' if live else 'local', 'kind': kind,
                'milestone_id': milestone_id, 'name': job['name'],
                'checkpoint_sha256': checkpoint_hash, 'feature_version': FEATURE_VERSION,
                'encoder_sha256': hashlib.sha256(files['bc_features.py']).hexdigest(),
                'main_sha256': hashlib.sha256(files['main.py']).hexdigest()}
    if 'ppo_context.py' in files:
        identity['context_sha256'] = hashlib.sha256(files['ppo_context.py']).hexdigest()
    snapshot_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity['snapshot_id'] = snapshot_id
    exports = safe_path(server.DATA, 'ppo_exports')
    exports.mkdir(parents=True, exist_ok=True)
    destination = safe_path(exports, snapshot_id)

    with EXPORT_LOCK:
        if destination.is_dir():
            artifact(server, snapshot_id, 'zip')
            artifact(server, snapshot_id, 'checkpoint')
            return json.loads((destination / 'report.json').read_text())
        temporary = Path(tempfile.mkdtemp(prefix='.preparing-', dir=exports))
        try:
            (temporary / 'full.pt').write_bytes(checkpoint_bytes)
            # Only our own export directory receives requests, weights, or reports.
            request = {'identity': identity, 'source_metadata': source_metadata}
            cloning.write_json(temporary / 'export-request.json', request)
            run_worker(temporary)
            report = json.loads((temporary / 'report.json').read_text())
            if report.get('snapshot_id') != snapshot_id:
                raise ValueError('The snapshot worker returned a different artifact')
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    artifact(server, snapshot_id, 'zip')
    artifact(server, snapshot_id, 'checkpoint')
    return report


def artifact(server, snapshot_id, format='zip'):
    if not isinstance(snapshot_id, str) or not re.fullmatch(r'[a-f0-9]{64}', snapshot_id):
        raise ValueError('Invalid PPO snapshot identifier')
    if format not in ('zip', 'checkpoint'):
        raise ValueError('Choose zip or checkpoint format')
    folder = safe_path(server.DATA, 'ppo_exports', snapshot_id)
    report_path = safe_path(folder, 'report.json')
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError('The prepared snapshot could not be found') from error
    if report.get('snapshot_id') != snapshot_id or report.get('export_version') != EXPORT_VERSION:
        raise ValueError('The prepared snapshot identity is invalid')
    filename = 'bot.zip' if format == 'zip' else 'full.pt'
    path = safe_path(folder, filename)
    digest = hashlib.sha256(stable_bytes(path)).hexdigest()
    expected = report.get('zip_sha256') if format == 'zip' else report.get('checkpoint_sha256')
    if digest != expected:
        raise ValueError('The prepared snapshot checksum changed; export it again')
    if format == 'zip' and not report.get('zip_available'):
        raise ValueError('This snapshot has not passed playable-bot validation')
    return path
