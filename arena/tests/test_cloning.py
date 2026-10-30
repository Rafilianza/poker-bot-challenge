"""Cloning checks: information boundaries, grouped splits, recovery and export."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from arena import cloning, server
from arena import bc_train_worker as training
from arena.bc_features import FEATURE_COUNT, action_options, action_target
from arena.bc_features import choose_action, encode_observation, forward_logits


def observation(size=4, to_call=2):
    return {'type': 'act', 'hand': 0, 'seat': 3, 'street': 'preflop',
            'hole': ['As', 'Kd'], 'board': [], 'pot': 3, 'to_call': to_call,
            'min_raise_to': 4, 'max_raise_to': 200, 'can_raise': True,
            'stacks': [200, 199, 198, 200] + [200] * (size - 4),
            'street_bets': [0, 1, 2, 0] + [0] * (size - 4),
            'folded': [False] * size, 'button': 0, 'history': [],
            'players': list(range(size)), 'clock_ms': 30100}


class FeatureTests(unittest.TestCase):
    def test_features_use_only_available_information(self):
        original = observation()
        encoded = encode_observation(original)
        self.assertEqual(len(encoded), FEATURE_COUNT)
        changed = deepcopy(original)
        changed.update(clock_ms=1, hand=100, players=[9, 8, 7, 6],
                       future_board=['2s', '3s', '4s'], opponent_hole=['Ah', 'Ad'])
        self.assertEqual(encoded, encode_observation(changed))
        changed['hole'] = ['2s', '3s']
        self.assertNotEqual(encoded, encode_observation(changed))

    def test_action_targets_and_masks_respect_engine_rules(self):
        state = observation()
        options, mask = action_options(state)
        self.assertEqual(action_target(state, {'action': 'raise', 'amount': 200}), (6, 0.0))
        for action_index, allowed in enumerate(mask):
            if allowed:
                target, error = action_target(state, options[action_index])
                self.assertEqual((target, error), (action_index, 0.0))
        state.update(min_raise_to=200)
        _, mask = action_options(state)
        self.assertEqual(mask, [True, True, False, False, False, False, True])
        state.update(can_raise=False, to_call=0)
        _, mask = action_options(state)
        self.assertEqual(mask, [False, True, False, False, False, False, False])
        self.assertEqual(choose_action(state, [100, -100, 100, 100, 100, 100, 100])['action'], 'check')
        with self.assertRaises(ValueError):
            action_target(state, {'action': 'raise', 'amount': 200})

    def test_raise_error_is_measured_in_actual_chips(self):
        state = observation()
        target, error = action_target(state, {'action': 'raise', 'amount': 30})
        options, mask = action_options(state)
        self.assertTrue(mask[target])
        self.assertEqual(error, abs(options[target]['amount'] - 30))
        with self.assertRaises(ValueError):
            action_target(state, {'action': 'raise', 'amount': 300})

    def test_exported_logits_match_torch(self):
        torch.manual_seed(11)
        model = training.build_model()
        for size in (4, 5, 6):
            features = encode_observation(observation(size))
            expected = model(torch.tensor(features)).detach().numpy()
            actual = forward_logits(features, training.model_layers(model))
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_seed_splits_are_global_and_nonempty(self):
        groups = ['seed-' + str(index) for index in range(23)]
        split = training.split_seed_groups(groups)
        self.assertEqual(set(split.values()), {0, 1, 2})
        self.assertEqual(split, training.split_seed_groups(groups[::-1] + groups))
        with self.assertRaisesRegex(ValueError, '3 distinct seed groups'):
            training.split_seed_groups(['a', 'b'])


class TrainingTests(unittest.TestCase):
    def test_explicit_cuda_is_validated_before_training(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA is unavailable'):
                training.get_device('cuda')
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(training.get_device('cuda').type, 'cuda')

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_data = server.DATA
        server.DATA = Path(self.temporary_directory.name)
        server.STOP.clear()
        server.init()
        with server.connect() as database:
            database.execute('CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            database.execute('CREATE TABLE IF NOT EXISTS training_jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.teacher = {'id': 'teacher-a', 'name': 'Teacher A', 'hash': 'version-a'}
        self.other_teacher = {'id': 'teacher-b', 'name': 'Teacher B', 'hash': 'version-b'}

    def tearDown(self):
        server.DATA = self.previous_data
        server.STOP.clear()
        self.temporary_directory.cleanup()

    def records(self, seed_count=6, per_seed=30):
        records = []
        for group in range(seed_count):
            for decision in range(per_seed):
                teacher = self.teacher if decision % 2 else self.other_teacher
                state = observation(4 + decision % 3)
                if decision % 2:
                    state['hole'] = ['2s', '7d']
                    action = {'action': 'fold', 'amount': 0}
                else:
                    action = {'action': 'call', 'amount': 2}
                records.append({'schema_version': 1, 'decision_id': f'decision-{group}-{decision}',
                                'seed_group': 'seed-' + str(group), 'teacher': teacher,
                                'observation': state, 'executed_action': action,
                                'verdict': 'OK', 'table_size': len(state['stacks'])})
        return records

    def dataset(self, key='dataset-a', records=None):
        if records is None:
            records = self.records()
        folder = server.DATA / 'datasets' / key
        folder.mkdir(parents=True)
        shard_path = folder / 'part-000001.jsonl.gz'
        with gzip.open(shard_path, 'wt') as shard:
            for record in records:
                shard.write(json.dumps(record) + '\n')
        manifest = {'schema_version': 1, 'id': key, 'name': key, 'records': len(records),
                    'teachers': [self.teacher, self.other_teacher],
                    'files': [{'name': shard_path.name, 'records': len(records),
                               'sha256': hashlib.sha256(shard_path.read_bytes()).hexdigest()}]}
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        dataset = {'id': key, 'name': key, 'status': 'ready', 'records': len(records),
                   'teachers': manifest['teachers'], 'seed_groups': sorted({record['seed_group'] for record in records})}
        server.save('datasets', dataset)
        return dataset

    def queue(self, dataset_ids=None, **settings):
        return cloning.queue(server, {'name': 'Student', 'dataset_ids': dataset_ids or ['dataset-a'],
                                      'epochs': 4, 'batch_size': 32, 'device': 'cpu', **settings})

    def request(self, job):
        folder = server.DATA / 'training_runs' / job['id']
        return json.loads((folder / 'request.json').read_text()), folder

    def test_preview_uses_one_cpu_thread_and_rejects_legacy_gpu_requests(self):
        self.dataset()
        with patch.object(server, 'DEVELOPMENT', True):
            job = self.queue(device='auto')
            self.assertEqual(job['config']['device'], 'cpu')
            self.assertEqual(job['config']['cpu_threads'], 1)
            with self.assertRaisesRegex(ValueError, 'CPU'):
                self.queue(device='mps')
        with patch.object(server, 'DEVELOPMENT', False):
            legacy = self.queue(device='auto')
        request, folder = self.request(legacy)
        before = (folder / 'request.json').read_bytes()
        with patch.object(server, 'DEVELOPMENT', True), patch.object(cloning.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(ValueError, 'CPU'):
                cloning.execute(server, legacy)
        process.assert_not_called()
        self.assertEqual((folder / 'request.json').read_bytes(), before)
        self.assertEqual(server.get('training_jobs', legacy['id']), legacy)

    def test_streaming_preparation_deduplicates_and_preserves_seed_groups(self):
        self.dataset()
        self.dataset('dataset-b')
        job = self.queue(['dataset-a', 'dataset-b'])
        request, folder = self.request(job)
        array, metadata = training.prepare_data(request, folder, {})
        self.assertEqual(metadata['records'], 180)
        self.assertEqual(metadata['duplicate_records'], 180)
        self.assertEqual(sum(metadata['split_counts'].values()), 180)
        for group in np.unique(array['group']):
            self.assertEqual(len(np.unique(array['split'][array['group'] == group])), 1)
        self.assertIsInstance(array, np.memmap)

    def test_filters_apply_before_global_split(self):
        self.dataset()
        job = self.queue(teachers=['teacher-a'], table_sizes=[4])
        request, folder = self.request(job)
        array, metadata = training.prepare_data(request, folder, {})
        self.assertEqual(metadata['records'], 30)
        self.assertEqual(metadata['teacher_ids'], ['teacher-a'])
        self.assertEqual(set(array['size']), {4})
        self.assertTrue(all(metadata['split_counts'].values()))

    def test_conflicting_duplicates_and_bad_checksums_are_rejected(self):
        self.dataset()
        changed = self.records()
        changed[0]['executed_action'] = {'action': 'fold', 'amount': 0}
        self.dataset('dataset-b', changed)
        job = self.queue(['dataset-a', 'dataset-b'])
        request, folder = self.request(job)
        with self.assertRaisesRegex(ValueError, 'Conflicting copies'):
            training.prepare_data(request, folder, {})
        other = self.queue()
        request, folder = self.request(other)
        shard = server.DATA / 'datasets' / 'dataset-a' / 'part-000001.jsonl.gz'
        with shard.open('ab') as destination:
            destination.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum failed'):
            training.prepare_data(request, folder, {})

    def test_queue_validates_before_saving(self):
        self.dataset()
        for settings in ({'epochs': 0}, {'batch_size': 1}, {'learning_rate': float('nan')},
                         {'teachers': []}, {'table_sizes': [3]}, {'balanced': 'yes'},
                         {'teacher_weights': {'teacher-a': 0}}, {'device': 'unknown'}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self.queue(**settings)
        self.assertEqual(server.all_rows('training_jobs'), [])

    def test_pause_resume_replays_unfinished_epoch_with_same_final_weights(self):
        self.dataset()
        reference = self.queue(epochs=3)
        request, folder = self.request(reference)
        report = {'status': 'running', 'history': []}
        training.train(request, folder, report)
        expected = json.loads((folder / 'weights.json').read_text())
        paused = self.queue(epochs=3)
        request, folder = self.request(paused)
        original_write = training.write_json
        def pause_after_first_epoch(path, value):
            original_write(path, value)
            if path.name == 'status.json' and value.get('epoch') == 1:
                original_write(folder / 'control.json', {'action': 'pause'})
        with patch.object(training, 'write_json', side_effect=pause_after_first_epoch):
            with self.assertRaises(training.TrainingPaused):
                training.train(request, folder, {'status': 'running', 'history': []})
        checkpoint = torch.load(folder / 'latest.pt', weights_only=False)
        self.assertEqual(checkpoint['epoch'], 1)
        original_write(folder / 'control.json', {'action': 'run'})
        training.train(request, folder, {'status': 'running', 'history': []})
        actual = json.loads((folder / 'weights.json').read_text())
        self.assertEqual(actual, expected)

    def test_full_worker_learns_and_exported_student_runs_at_every_size(self):
        self.dataset()
        job = self.queue(epochs=12)
        completed = cloning.execute(server, job)
        self.assertEqual(completed['status'], 'completed', completed.get('error'))
        self.assertEqual(completed['epoch'], 12)
        self.assertLess(completed['history'][-1]['validation_loss'], completed['history'][0]['validation_loss'])
        self.assertGreater(completed['metrics']['test']['accuracy'], 0.8)
        self.assertEqual(set(completed['metrics']['test']['by_table_size']), {'4', '5', '6'})
        agent = cloning.export_agent(server, job['id'])
        self.assertEqual(cloning.export_agent(server, job['id'])['id'], agent['id'])
        exported = server.get('training_jobs', job['id'])
        self.assertEqual(len(exported['export_validation']), 5)
        for result in exported['export_validation']:
            self.assertEqual(result['verdicts'], ['OK'] * result['table_size'])
        source = (server.DATA / 'agents' / agent['id'] / 'main.py').read_text()
        self.assertNotIn('import torch', source)
        self.assertNotIn('import numpy', source)

    def test_terminal_status_is_hidden_until_worker_process_exits(self):
        self.dataset()
        job = self.queue()
        _, folder = self.request(job)
        observed_statuses = []
        class FinishingProcess:
            calls = 0
            def __init__(self, *args, **kwargs):
                cloning.write_json(folder / 'status.json', {'status': 'paused', 'phase': 'Paused'})
            def poll(self):
                self.calls += 1
                if self.calls == 1:
                    return None
                observed_statuses.append(server.get('training_jobs', job['id'])['status'])
                return 0
        with patch.object(cloning.subprocess, 'Popen', FinishingProcess):
            with patch.object(server, 'kill_tree'):
                result = cloning.execute(server, job)
        self.assertEqual(observed_statuses, ['running'])
        self.assertEqual(result['status'], 'paused')

    def test_pause_control_and_failed_job_resume(self):
        self.dataset()
        job = self.queue()
        self.assertEqual(cloning.control(server, job['id'], 'pause')['status'], 'paused')
        self.assertEqual(cloning.control(server, job['id'], 'resume')['status'], 'queued')
        with self.assertRaises(ValueError):
            cloning.control(server, job['id'], 'resume')
        job['status'] = 'failed'
        server.save('training_jobs', job)
        self.assertEqual(cloning.control(server, job['id'], 'resume')['status'], 'queued')


if __name__ == '__main__':
    unittest.main()
