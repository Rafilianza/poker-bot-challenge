"""Train one behaviour-cloning student in a separate, resumable process."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time
import traceback

from arena.bc_features import ACTION_NAMES, FEATURE_COUNT, FEATURE_VERSION
from arena.bc_features import action_options, action_target, encode_observation


class TrainingPaused(Exception):
    pass


class FinishRequested(Exception):
    pass


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def control_action(folder):
    path = folder / 'control.json'
    if path.exists():
        return json.loads(path.read_text()).get('action')
    return None


def check_pause(folder):
    if control_action(folder) == 'pause':
        raise TrainingPaused()


def check_training_control(folder):
    action = control_action(folder)
    if action == 'pause':
        raise TrainingPaused()
    # A finish request must never export the untrained epoch-zero model.
    # Without a validated checkpoint, complete the first epoch first.
    if action == 'finish' and (folder / 'best.pt').exists():
        raise FinishRequested()


def split_seed_groups(seed_groups):
    """Keep every related game together, even across multiple source datasets."""
    groups = sorted(set(seed_groups), key=lambda value: hashlib.sha256(
        ('bc-split-v1:' + value).encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError('Training needs at least 3 distinct seed groups after filtering. '
                         'Collect more batches or select more datasets.')
    held_out = max(1, round(len(groups) * 0.1))
    train_end = len(groups) - 2 * held_out
    return {group: 0 if index < train_end else 1 if index < train_end + held_out else 2
            for index, group in enumerate(groups)}


def prepare_data(request, folder, report):
    """Stream compressed demonstrations into a disk-backed numeric array."""
    import numpy as np

    config = request['config']
    selected_teachers = set(config['teachers'])
    selected_sizes = set(config['table_sizes'])
    dtype = np.dtype([
        ('features', '<f4', (FEATURE_COUNT,)), ('mask', '?', (len(ACTION_NAMES),)),
        ('target', 'u1'), ('teacher', '<u2'), ('size', 'u1'),
        ('group', '<u4'), ('split', 'u1'),
    ])
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    metadata_path = folder / 'prepared.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['request_hash'] == request_hash:
            array = np.memmap(folder / 'examples.bin', mode='r', dtype=dtype,
                              shape=(metadata['records'],))
            return array, metadata

    teacher_ids = sorted(selected_teachers)
    teacher_indices = {teacher: index for index, teacher in enumerate(teacher_ids)}
    group_indices = {}
    duplicate_records = 0
    records = 0
    raise_count = 0
    raise_error = 0.0
    maximum_raise_error = 0.0
    row_buffer = []
    database_path = folder / 'deduplication.sqlite'
    database_path.unlink(missing_ok=True)
    database = sqlite3.connect(database_path)
    database.execute('CREATE TABLE decisions (id TEXT PRIMARY KEY, digest TEXT NOT NULL)')
    report.update(phase='Preparing data', progress='Reading and checking dataset shards')
    write_json(folder / 'status.json', report)
    try:
        with (folder / 'examples.bin').open('wb') as destination:
            for dataset in request['datasets']:
                dataset_folder = Path(dataset['path'])
                manifest_path = dataset_folder / 'manifest.json'
                if file_hash(manifest_path) != dataset['manifest_sha256']:
                    raise ValueError('A selected dataset manifest changed after training was queued')
                manifest = json.loads(manifest_path.read_text())
                teacher_hashes = {teacher['id']: teacher.get('hash') for teacher in manifest['teachers']}
                for shard in manifest['files']:
                    check_pause(folder)
                    shard_path = dataset_folder / shard['name']
                    if shard_path.parent.resolve() != dataset_folder.resolve() or shard_path.is_symlink():
                        raise ValueError('Dataset contains an unsafe shard path')
                    if file_hash(shard_path) != shard['sha256']:
                        raise ValueError('Dataset checksum failed: ' + shard['name'])
                    shard_count = 0
                    with gzip.open(shard_path, 'rt', encoding='utf-8') as source:
                        for line in source:
                            record = json.loads(line)
                            shard_count += 1
                            observation = record['observation']
                            teacher_id = record['teacher']['id']
                            size = record.get('table_size', len(observation['stacks']))
                            if teacher_id not in selected_teachers or size not in selected_sizes:
                                continue
                            if record.get('verdict') != 'OK':
                                raise ValueError('Dataset contains a failed teacher decision')
                            if size != len(observation['stacks']):
                                raise ValueError('Recorded table size disagrees with the observation')
                            if teacher_hashes.get(teacher_id) != record['teacher'].get('hash'):
                                raise ValueError('Teacher version does not match the dataset manifest')
                            seed_group = record['seed_group']
                            if not isinstance(seed_group, str) or not seed_group:
                                raise ValueError('A decision has no valid seed group')
                            decision_id = record['decision_id']
                            digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                            try:
                                database.execute('INSERT INTO decisions VALUES (?,?)', (decision_id, digest))
                            except sqlite3.IntegrityError:
                                previous = database.execute('SELECT digest FROM decisions WHERE id=?',
                                                            (decision_id,)).fetchone()[0]
                                if previous != digest:
                                    raise ValueError('Conflicting copies of decision ' + decision_id)
                                duplicate_records += 1
                                continue
                            if seed_group not in group_indices:
                                group_indices[seed_group] = len(group_indices)
                            target, approximation_error = action_target(observation, record['executed_action'])
                            _, mask = action_options(observation)
                            features = encode_observation(observation)
                            if not all(np.isfinite(features)):
                                raise ValueError('A demonstration contains non-finite input values')
                            row_buffer.append((features, mask, target, teacher_indices[teacher_id],
                                               size, group_indices[seed_group], 0))
                            if record['executed_action']['action'] == 'raise':
                                raise_count += 1
                                raise_error += approximation_error
                                maximum_raise_error = max(maximum_raise_error, approximation_error)
                            records += 1
                            if len(row_buffer) >= 4096:
                                np.array(row_buffer, dtype=dtype).tofile(destination)
                                row_buffer.clear()
                                database.commit()
                                report.update(records=records, duplicate_records=duplicate_records,
                                              progress=f'Prepared {records:,} unique decisions')
                                write_json(folder / 'status.json', report)
                                check_pause(folder)
                    if shard_count != shard['records']:
                        raise ValueError('Dataset shard record count does not match its manifest')
            if row_buffer:
                np.array(row_buffer, dtype=dtype).tofile(destination)
    finally:
        database.close()
    if not records:
        raise ValueError('No demonstrations match the selected teachers and table sizes')
    split_by_group = split_seed_groups(group_indices)
    group_splits = np.array([split_by_group[group] for group in group_indices], dtype='u1')
    array = np.memmap(folder / 'examples.bin', mode='r+', dtype=dtype, shape=(records,))
    for start in range(0, records, 65536):
        array['split'][start:start + 65536] = group_splits[array['group'][start:start + 65536]]
    array.flush()
    split_names = ['train', 'validation', 'test']
    split_counts = {name: int(np.sum(array['split'] == index))
                    for index, name in enumerate(split_names)}
    if not all(split_counts.values()):
        raise ValueError('A train/validation/test split is empty after filtering')
    metadata = {
        'request_hash': request_hash, 'feature_version': FEATURE_VERSION,
        'records': records, 'duplicate_records': duplicate_records,
        'teacher_ids': teacher_ids, 'split_counts': split_counts,
        'seed_groups': {group: split_names[split] for group, split in split_by_group.items()},
        'raise_approximation': {'raises': raise_count, 'mean_absolute_chips': raise_error / max(1, raise_count),
                                'mean_absolute_bb': raise_error / max(1, raise_count) / 2.0,
                                'maximum_absolute_chips': maximum_raise_error},
    }
    write_json(metadata_path, metadata)
    database_path.unlink(missing_ok=True)
    return array, metadata


def build_model():
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_COUNT, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, len(ACTION_NAMES)),
    )


def model_layers(model):
    import torch
    layers = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            layers.append({'weight': layer.weight.detach().cpu().tolist(),
                           'bias': layer.bias.detach().cpu().tolist()})
    return layers


def get_device(preference):
    import torch
    if preference == 'cpu':
        return torch.device('cpu')
    if preference == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or choose CPU.')
        return torch.device('cuda')
    if preference == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError('The Apple GPU is unavailable. Choose Auto or CPU.')
        return torch.device('mps')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def batch_tensors(array, indices, device):
    import numpy as np
    import torch
    rows = array[indices]
    features = torch.from_numpy(np.array(rows['features'])).to(device)
    masks = torch.from_numpy(np.array(rows['mask'])).to(device)
    targets = torch.from_numpy(np.array(rows['target'], dtype=np.int64)).to(device)
    return features, masks, targets, rows


def evaluate(model, array, indices, batch_size, device, teacher_ids, folder, allow_finish=False):
    import torch
    totals = Counter(records=0, correct=0, loss=0.0)
    by_teacher = {}
    by_size = {}
    by_teacher_size = {}
    confusion = [[0] * len(ACTION_NAMES) for _ in ACTION_NAMES]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            if allow_finish:
                check_training_control(folder)
            else:
                check_pause(folder)
            features, masks, targets, rows = batch_tensors(array, indices[start:start + batch_size], device)
            logits = model(features).masked_fill(~masks, -1e9)
            losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none').cpu().tolist()
            predictions = logits.argmax(dim=1).cpu().tolist()
            for row, prediction, loss in zip(rows, predictions, losses):
                target = int(row['target'])
                correct = int(prediction == target)
                teacher = teacher_ids[int(row['teacher'])]
                size = str(int(row['size']))
                confusion[target][prediction] += 1
                for group, key in ((by_teacher, teacher), (by_size, size),
                                   (by_teacher_size, teacher + ':' + size)):
                    counter = group.setdefault(key, Counter(records=0, correct=0, loss=0.0))
                    counter.update(records=1, correct=correct, loss=loss)
                totals.update(records=1, correct=correct, loss=loss)
    def finish(counter):
        count = counter['records']
        return {'records': count, 'loss': counter['loss'] / count,
                'accuracy': counter['correct'] / count}
    result = finish(totals)
    result['by_teacher'] = {key: finish(value) for key, value in by_teacher.items()}
    result['by_table_size'] = {key: finish(value) for key, value in by_size.items()}
    result['by_teacher_size'] = {key: finish(value) for key, value in by_teacher_size.items()}
    result['confusion'] = confusion
    return result


def save_checkpoint(path, model, optimizer, epoch, history, best_loss):
    import torch
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'epoch': epoch, 'history': history, 'best_loss': best_loss,
             'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}
    device = next(model.parameters()).device
    if device.type == 'mps':
        state['mps_rng'] = torch.mps.get_rng_state()
    if device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train(request, folder, report):
    import numpy as np
    import torch
    # Small tabular batches are faster and leave the dashboard responsive with
    # a few CPU threads instead of the machine's full thread pool.
    torch.set_num_threads(min(request['config'].get('cpu_threads', 4), torch.get_num_threads()))
    array, metadata = prepare_data(request, folder, report)
    report.update({key: metadata[key] for key in ('records', 'duplicate_records', 'split_counts',
                                                'raise_approximation')})
    config = request['config']
    device = get_device(config['device'])
    report.update(device=str(device), phase='Training')
    random.seed(4217)
    torch.manual_seed(4217)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    checkpoint_path = folder / 'latest.pt'
    history = []
    completed_epochs = 0
    best_loss = float('inf')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        completed_epochs = checkpoint['epoch']
        history = checkpoint['history']
        best_loss = checkpoint['best_loss']
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng'])
        if device.type == 'mps' and 'mps_rng' in checkpoint:
            torch.mps.set_rng_state(checkpoint['mps_rng'])
        if device.type == 'cuda' and 'cuda_rng' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    else:
        save_checkpoint(checkpoint_path, model, optimizer, 0, history, best_loss)
    validation_indices = np.flatnonzero(array['split'] == 1)
    test_indices = np.flatnonzero(array['split'] == 2)
    teacher_ids = metadata['teacher_ids']
    finish_now = control_action(folder) == 'finish' and (folder / 'best.pt').exists()
    if control_action(folder) == 'finish':
        report['finish_requested'] = True
    # Resuming only to finalize does not need another scan of the training rows.
    if not finish_now:
        train_indices = np.flatnonzero(array['split'] == 0)
        train_counts = Counter()
        for index in train_indices:
            row = array[index]
            train_counts[(int(row['teacher']), int(row['size']))] += 1
        weights = {}
        for key, count in train_counts.items():
            teacher_id = teacher_ids[key[0]]
            teacher_weight = config['teacher_weights'].get(teacher_id, 1.0)
            weights[key] = teacher_weight / count if config['balanced'] else teacher_weight
        normalizer = sum(weights[key] * count for key, count in train_counts.items()) / len(train_indices)
        weights = {key: value / normalizer for key, value in weights.items()}
    report.update(epoch=completed_epochs, epochs=config['epochs'], history=history,
                  progress=f'Completed {completed_epochs} of {config["epochs"]} epochs')
    write_json(folder / 'status.json', report)
    try:
        if finish_now:
            raise FinishRequested()
        for epoch in range(completed_epochs + 1, config['epochs'] + 1):
            check_training_control(folder)
            model.train()
            order = torch.randperm(len(train_indices)).numpy()
            total_loss = 0.0
            total_correct = 0
            seen = 0
            last_report = 0.0
            for start in range(0, len(order), config['batch_size']):
                check_training_control(folder)
                batch_indices = train_indices[order[start:start + config['batch_size']]]
                features, masks, targets, rows = batch_tensors(array, batch_indices, device)
                batch_weights = [weights[(int(row['teacher']), int(row['size']))] for row in rows]
                batch_weights = torch.tensor(batch_weights, dtype=torch.float32, device=device)
                optimizer.zero_grad()
                logits = model(features).masked_fill(~masks, -1e9)
                losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
                loss = (losses * batch_weights).mean()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(rows)
                total_correct += int((logits.argmax(dim=1) == targets).sum().cpu())
                seen += len(rows)
                if time.monotonic() - last_report > 0.75:
                    report.update(progress=f'Epoch {epoch}/{config["epochs"]}: {seen:,}/{len(order):,} training decisions',
                                  batch_progress=seen / len(order))
                    write_json(folder / 'status.json', report)
                    last_report = time.monotonic()
            validation = evaluate(model, array, validation_indices, config['batch_size'], device, teacher_ids, folder, allow_finish=True)
            history.append({'epoch': epoch, 'train_loss': total_loss / seen,
                            'train_accuracy': total_correct / seen,
                            'validation_loss': validation['loss'], 'validation_accuracy': validation['accuracy']})
            if validation['loss'] < best_loss:
                best_loss = validation['loss']
                save_checkpoint(folder / 'best.pt', model, optimizer, epoch, history, best_loss)
            save_checkpoint(checkpoint_path, model, optimizer, epoch, history, best_loss)
            completed_epochs = epoch
            report.update(epoch=epoch, history=history, metrics={'validation': validation},
                          progress=f'Completed {epoch} of {config["epochs"]} epochs', batch_progress=1.0)
            write_json(folder / 'status.json', report)
    except FinishRequested:
        # Discard changes and metrics from an unfinished epoch. The latest
        # checkpoint is the last complete epoch, even if the live model moved on.
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        completed_epochs = saved['epoch']
        history = saved['history']
        report['finish_requested'] = True
    # Evaluate and export the checkpoint chosen using validation data only.
    best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    if type(best.get('epoch')) is not int or best['epoch'] < 1:
        raise ValueError('Finish requires a checkpoint from at least one fully validated epoch')
    # Saving best.pt precedes latest.pt. After a crash between those writes,
    # the validated best checkpoint may contain one more complete epoch.
    if best['epoch'] > completed_epochs:
        completed_epochs = best['epoch']
        history = best['history']
    model.load_state_dict(best['model'])
    report.update(epoch=completed_epochs, epochs=config['epochs'], history=history,
                  stopped_early=completed_epochs < config['epochs'], best_epoch=best['epoch'],
                  phase='Finalizing best checkpoint',
                  progress=f'Validating and testing the best checkpoint from epoch {best["epoch"]}')
    write_json(folder / 'status.json', report)
    # Finish no longer interrupts evaluation, but Pause and server shutdown do.
    validation = evaluate(model, array, validation_indices, config['batch_size'], device, teacher_ids, folder)
    test = evaluate(model, array, test_indices, config['batch_size'], device, teacher_ids, folder)
    weights_payload = {'schema_version': 1, 'feature_version': FEATURE_VERSION,
                       'feature_count': FEATURE_COUNT, 'action_names': ACTION_NAMES,
                       'layers': model_layers(model), 'best_epoch': best['epoch']}
    write_json(folder / 'weights.json', weights_payload)
    report.update(status='completed', phase='Completed', progress='Student ready to export',
                  metrics={'validation': validation, 'test': test}, best_epoch=best['epoch'],
                  finish_requested=False)
    write_json(folder / 'status.json', report)
    write_json(folder / 'result.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    request_path = Path(args.request)
    folder = request_path.parent
    report = {'status': 'running', 'phase': 'Starting', 'epoch': 0,
              'epochs': 0, 'history': [], 'progress': 'Loading training dependencies'}
    try:
        request = json.loads(request_path.read_text())
        report['epochs'] = request['config']['epochs']
        write_json(folder / 'status.json', report)
        train(request, folder, report)
    except TrainingPaused:
        report.update(status='paused', phase='Paused',
                      progress='Paused. Resume continues from the last completed epoch; an unfinished epoch is replayed.')
        write_json(folder / 'status.json', report)
    except Exception as error:
        report.update(status='failed', phase='Failed', error=str(error), progress='Training stopped')
        write_json(folder / 'status.json', report)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
