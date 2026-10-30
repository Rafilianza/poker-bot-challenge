"""Read-only saved-actor exports, atomic source reads, and SDK playable ZIPs."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import torch

from arena import cloning, ppo_snapshots
from arena import ppo_snapshot_worker
from arena.bc_features import FEATURE_VERSION, encode_observation, forward_logits
from arena.bc_train_worker import build_model


def checkpoint_bytes(episodes=50, iteration=5):
    actor = build_model()
    checkpoint = {
        'actor': actor.state_dict(), 'critic': {'saved': 'critic'},
        'actor_optimizer': {'saved': 'actor optimizer'},
        'critic_optimizer': {'saved': 'critic optimizer'},
        'torch_rng': torch.get_rng_state(), 'python_rng': ('saved',),
        'request_hash': 'original-request', 'feature_version': FEATURE_VERSION,
        'progress': {'episodes': episodes, 'iteration': iteration, 'best_iteration': iteration,
                     'best_score': 12.5, 'validation': {'bb100': 12.5, 'hands': 12}},
    }
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    return stream.getvalue()


def tree_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()}


class PPOSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.data = root / 'preview'
        self.live = root / 'main'
        self.data.mkdir()
        self.live.mkdir()
        self.key = 'a' * 32
        self.job = {'id': self.key, 'name': 'Saved PPO', 'status': 'paused',
                    'episodes': 999, 'iteration': 99, 'target_episodes': 1000}
        self.folder = self.data / 'ppo_runs' / self.key
        self.folder.mkdir(parents=True)
        self.saved = checkpoint_bytes()
        (self.folder / 'best.pt').write_bytes(self.saved)
        (self.folder / 'latest.pt').write_bytes(checkpoint_bytes(900, 90))
        (self.folder / 'request.json').write_text('{"original":true}')
        (self.folder / 'control.json').write_text('{"action":"pause"}')
        self.server = SimpleNamespace(DATA=self.data, LIVE_PPO_DATA=self.live,
                                      get=lambda table, key: deepcopy(self.job))

    def tearDown(self):
        self.temporary.cleanup()

    def fake_validation(self, folder):
        return [{'table_size': size, 'hands': 2, 'verdicts': ['OK'] * size} for size in range(2, 7)]

    def fast_worker(self, folder):
        with mock.patch.object(ppo_snapshot_worker, 'validate_games', side_effect=self.fake_validation):
            ppo_snapshot_worker.export(folder)

    def test_paused_best_produces_clean_playable_zip_without_any_job_changes(self):
        before = tree_hashes(self.folder)
        job_before = deepcopy(self.job)
        result = ppo_snapshots.prepare(self.server, self.key)
        self.assertEqual(result['episodes'], 50)
        self.assertEqual(result['iteration'], 5)
        self.assertEqual(result['score'], 12.5)
        self.assertEqual(result['selection_metric'], 'bb100')
        self.assertEqual(result['device'], 'cpu')
        self.assertEqual(len(result['validation']), 5)
        self.assertTrue(all(set(check['verdicts']) == {'OK'} for check in result['validation']))
        self.assertEqual(tree_hashes(self.folder), before)
        self.assertEqual(self.job, job_before)
        self.assertFalse((self.data / 'agents').exists())
        saved_path = ppo_snapshots.artifact(self.server, result['snapshot_id'], 'checkpoint')
        self.assertEqual(saved_path.read_bytes(), self.saved)
        with zipfile.ZipFile(ppo_snapshots.artifact(self.server, result['snapshot_id'])) as archive:
            self.assertEqual(set(archive.namelist()), {'main.py', 'weights.json', 'bc_features.py'})
            self.assertNotIn('torch', archive.read('main.py').decode())
            self.assertNotIn('import torch', archive.read('bc_features.py').decode())
            model = json.loads(archive.read('weights.json'))
        actor = build_model()
        actor.load_state_dict(torch.load(io.BytesIO(self.saved), weights_only=False)['actor'])
        features = [0.0] * 189
        actual = forward_logits(features, model['layers'])
        expected = actor(torch.tensor(features)).detach().tolist()
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_cache_pins_identity_when_best_is_replaced(self):
        with mock.patch.object(ppo_snapshots, 'run_worker', side_effect=self.fast_worker) as worker:
            first = ppo_snapshots.prepare(self.server, self.key)
            second = ppo_snapshots.prepare(self.server, self.key)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(first['snapshot_id'], second['snapshot_id'])
            replacement = checkpoint_bytes(60, 6)
            (self.folder / 'best.tmp').write_bytes(replacement)
            (self.folder / 'best.tmp').replace(self.folder / 'best.pt')
            third = ppo_snapshots.prepare(self.server, self.key)
        self.assertNotEqual(first['snapshot_id'], third['snapshot_id'])
        self.assertEqual(third['episodes'], 60)
        self.assertEqual(ppo_snapshots.artifact(self.server, first['snapshot_id'], 'checkpoint').read_bytes(), self.saved)

    def test_atomic_rename_during_read_keeps_one_complete_checkpoint(self):
        path = self.folder / 'best.pt'
        replacement = self.folder / 'best.tmp'
        replacement.write_bytes(checkpoint_bytes(70, 7))
        original_fstat = os.fstat
        calls = 0

        def replace_after_open(descriptor):
            nonlocal calls
            information = original_fstat(descriptor)
            calls += 1
            if calls == 1:
                replacement.replace(path)
            return information

        with mock.patch.object(ppo_snapshots.os, 'fstat', side_effect=replace_after_open):
            captured = ppo_snapshots.stable_bytes(path)
        self.assertEqual(captured, self.saved)
        self.assertNotEqual(captured, path.read_bytes())

    def test_live_sqlite_source_and_training_files_remain_byte_identical(self):
        live_folder = self.live / 'ppo_runs' / self.key
        live_folder.mkdir(parents=True)
        (live_folder / 'best.pt').write_bytes(self.saved)
        (live_folder / 'request.json').write_text('{"running":true}')
        saved_job = dict(self.job, status='running')
        saved_job.update(
            config={'reward_mode': 'chips', 'training_mode': 'self_play'},
            history=[{'iteration': 99, 'episodes': 999, 'train_bb100': -2.5,
                      'validation_bb100': 12.5}],
            baseline={'validation': {'bb100': 4.0}},
            best_validation={'bb100': 12.5}, best_score=12.5, best_iteration=99,
            baseline_test={'bb100': 3.0}, test=None,
            agent_id='b' * 32, private_note='Not a chart field',
        )
        database = sqlite3.connect(self.live / 'arena.sqlite')
        database.execute('CREATE TABLE ppo_jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        database.execute('INSERT INTO ppo_jobs VALUES (?, ?)',
                         (self.key, json.dumps(saved_job)))
        database.commit()
        database.close()
        before = tree_hashes(self.live)
        rows = ppo_snapshots.list_live(self.server)
        self.assertEqual(rows[0]['status'], 'running')
        self.assertTrue(rows[0]['read_only'])
        self.assertTrue(rows[0]['best_available'])
        for field in ('config', 'history', 'baseline', 'best_validation', 'best_score',
                      'best_iteration', 'baseline_test', 'test'):
            self.assertEqual(rows[0][field], saved_job[field])
        self.assertNotIn('agent_id', rows[0])
        self.assertNotIn('private_note', rows[0])
        self.assertNotIn('latest_validation', rows[0])
        self.assertEqual(tree_hashes(self.live), before)
        with mock.patch.object(ppo_snapshots, 'run_worker', side_effect=self.fast_worker):
            result = ppo_snapshots.prepare(self.server, self.key, live=True)
        self.assertEqual(result['source'], 'live')
        self.assertEqual(result['episodes'], 50)
        self.assertEqual(tree_hashes(self.live), before)
        self.assertTrue(ppo_snapshots.artifact(self.server, result['snapshot_id']).is_relative_to(self.data.resolve()))

    def test_unavailable_invalid_checkpoint_and_unsafe_ids_cannot_export(self):
        for invalid in ('../anything', 'A' * 32, ''):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ppo_snapshots.prepare(self.server, invalid)
        for invalid in ('../anything', 'z' * 64, ''):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ppo_snapshots.artifact(self.server, invalid)
        (self.folder / 'best.pt').unlink()
        self.assertFalse(ppo_snapshots.availability(self.folder)['available'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            ppo_snapshots.prepare(self.server, self.key)
        (self.folder / 'best.pt').write_bytes(b'corrupt checkpoint')
        with mock.patch.object(ppo_snapshots, 'run_worker', side_effect=self.fast_worker):
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                ppo_snapshots.prepare(self.server, self.key)
        self.assertEqual(list((self.data / 'ppo_exports').iterdir()), [])

    def test_malformed_actor_and_feature_versions_are_rejected(self):
        original = torch.load(io.BytesIO(self.saved), weights_only=False)
        bad_checkpoints = []
        bad_version = deepcopy(original)
        bad_version['feature_version'] = -1
        bad_checkpoints.append((bad_version, 'feature version'))
        bad_shape = deepcopy(original)
        bad_shape['actor']['0.weight'] = torch.zeros((1, 1))
        bad_checkpoints.append((bad_shape, 'dimensions'))
        bad_values = deepcopy(original)
        bad_values['actor']['0.weight'][0][0] = float('nan')
        bad_checkpoints.append((bad_values, 'non-finite'))
        for checkpoint, error in bad_checkpoints:
            stream = io.BytesIO()
            torch.save(checkpoint, stream)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                ppo_snapshot_worker.load_actor(stream.getvalue())

    def test_cached_artifact_checksums_and_paths_are_checked(self):
        with mock.patch.object(ppo_snapshots, 'run_worker', side_effect=self.fast_worker):
            result = ppo_snapshots.prepare(self.server, self.key)
        artifact = ppo_snapshots.artifact(self.server, result['snapshot_id'])
        artifact.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            ppo_snapshots.artifact(self.server, result['snapshot_id'])
        artifact.unlink()
        artifact.symlink_to(self.folder / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'inside'):
            ppo_snapshots.artifact(self.server, result['snapshot_id'])

    def test_shared_export_source_remains_valid_python(self):
        files = cloning.playable_files('Name with "quotes"', b'{}')
        compile(files['main.py'], 'main.py', 'exec')
        self.assertEqual(set(files), {'main.py', 'weights.json', 'bc_features.py'})

    def test_polling_only_stats_best_and_does_not_deserialize_actor(self):
        with mock.patch.object(ppo_snapshots, 'stable_bytes', side_effect=AssertionError('checkpoint read')):
            with mock.patch.object(ppo_snapshots, 'run_worker', side_effect=AssertionError('worker started')):
                result = ppo_snapshots.describe(self.server, self.key)
        self.assertTrue(result['best']['available'])
        self.assertNotIn('episodes', result['best'])

    def test_archived_baseline_and_ppo_export_the_exact_saved_policy(self):
        from arena import ppo_milestones
        source = self.folder / 'source_bc.pt'
        torch.save({'model': build_model().state_dict(), 'epoch': 2}, source)
        request = {'job_id': self.key, 'config': {'seed': 42, 'target_episodes': 100},
                   'opponents': [], 'source_checkpoint': {'path': str(source),
                       'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}
        (self.folder / 'request.json').write_text(json.dumps(request))
        ppo_milestones.ensure_settings(self.folder, request)
        baseline = ppo_milestones.archive(self.folder, request, source, 'baseline', 0, 0)
        saved = torch.load(io.BytesIO(self.saved), weights_only=False)
        saved['request_hash'] = ppo_milestones.request_hash(request)
        torch.save(saved, self.folder / 'latest.pt')
        milestone = ppo_milestones.archive(self.folder, request, self.folder / 'latest.pt', 'milestone', 50, 5)
        before = tree_hashes(self.folder)
        with mock.patch.object(ppo_snapshots, 'run_worker', side_effect=self.fast_worker):
            initial = ppo_snapshots.prepare(self.server, self.key, kind='milestone', milestone_id=baseline['id'])
            trained = ppo_snapshots.prepare(self.server, self.key, kind='milestone', milestone_id=milestone['id'])
        self.assertEqual(initial['episodes'], 0)
        self.assertEqual(initial['checkpoint_format'], 'bc')
        self.assertFalse(initial['resumable'])
        self.assertEqual(trained['episodes'], 50)
        self.assertEqual(trained['checkpoint_sha256'], milestone['checkpoint_sha256'])
        self.assertTrue(trained['resumable'])
        self.assertEqual(tree_hashes(self.folder), before)


if __name__ == '__main__':
    unittest.main()
