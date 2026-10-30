"""Actor-only GRPO downloads preserve training state and scaffold compatibility."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from arena import cloning, grpo_snapshots, grpo_snapshot_worker, ppo_milestones, ppo_snapshot_worker
from arena.bc_features import FEATURE_VERSION
from arena.bc_train_worker import build_model
from arena.ppo_context import CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT
from arena.ppo_models import build_actor


def file_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()}


class GRPOSnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name).resolve()
        self.key = 'd' * 32
        self.folder = self.data / 'grpo_runs' / self.key
        self.folder.mkdir(parents=True)
        self.config = {'algorithm': 'grpo', 'reward_mode': 'game_placement', 'placement_weight': 0.9,
                       'chip_scale_bb100': 100, 'seed': 42, 'target_episodes': 10000}
        self.job = {'id': self.key, 'name': 'GRPO learner', 'status': 'running', 'config': self.config}

        def get(table, key):
            self.assertEqual(table, 'grpo_jobs')
            self.assertEqual(key, self.key)
            return deepcopy(self.job)

        self.server = SimpleNamespace(DATA=self.data, get=get)
        source = self.folder / 'source_bc.pt'
        torch.save({'model': build_model().state_dict(), 'epoch': 1}, source)
        self.request = {'job_id': self.key, 'feature_version': FEATURE_VERSION, 'config': self.config,
                        'source_checkpoint': {'path': str(source), 'sha256': ppo_milestones.file_hash(source)},
                        'opponents': []}
        cloning.write_json(self.folder / 'request.json', self.request)
        cloning.write_json(self.folder / 'control.json', {'action': 'pause'})
        self.checkpoint = {
            'algorithm': 'grpo', 'actor': build_actor(context=True).state_dict(), 'actor_optimizer': {},
            'torch_rng': torch.get_rng_state(), 'python_rng': (),
            'request_hash': ppo_milestones.request_hash(self.request),
            'feature_version': FEATURE_VERSION, 'feature_count': PLACEMENT_FEATURE_COUNT,
            'context_version': CONTEXT_VERSION, 'reward_mode': 'game_placement', 'reward_config': self.config,
            'selection_metric': 'placement_chip_score',
            'progress': {'episodes': 400, 'iteration': 1, 'best_iteration': 1, 'best_score': 0.3,
                         'selection_metric': 'placement_chip_score',
                         'validation': {'selection_score': 0.3, 'bb100': 12.5}},
        }
        torch.save(self.checkpoint, self.folder / 'best.pt')
        latest = deepcopy(self.checkpoint)
        latest['progress']['episodes'] = 800
        latest['progress']['iteration'] = 2
        torch.save(latest, self.folder / 'latest.pt')

    def tearDown(self):
        self.temporary.cleanup()

    def fast_worker(self, folder):
        validation = [{'table_size': size, 'hands': 2, 'verdicts': ['OK'] * size} for size in range(2, 7)]
        with patch.object(ppo_snapshot_worker, 'validate_games', return_value=validation):
            return grpo_snapshot_worker.export(folder)

    def test_real_sdk_best_zip_and_critic_free_checkpoint_do_not_change_training(self):
        before = file_hashes(self.folder)
        result = grpo_snapshots.prepare(self.server, self.key)
        self.assertEqual(result['algorithm'], 'grpo')
        self.assertEqual(result['checkpoint_format'], 'grpo')
        self.assertTrue(result['resumable'])
        self.assertEqual(result['episodes'], 400)
        self.assertEqual(result['score'], 0.3)
        self.assertEqual(result['selection_metric'], 'placement_chip_score')
        self.assertEqual(result['metadata']['algorithm'], 'grpo')
        self.assertTrue(all(set(check['verdicts']) == {'OK'} for check in result['validation']))
        self.assertEqual(file_hashes(self.folder), before)
        self.assertFalse((self.data / 'ppo_exports').exists())
        self.assertFalse((self.data / 'agents').exists())
        checkpoint = grpo_snapshots.artifact(self.server, result['snapshot_id'], 'pt')
        self.assertEqual(checkpoint.read_bytes(), (self.folder / 'best.pt').read_bytes())
        self.assertEqual(checkpoint, grpo_snapshots.artifact(self.server, result['snapshot_id'], 'checkpoint'))
        with zipfile.ZipFile(grpo_snapshots.artifact(self.server, result['snapshot_id'])) as archive:
            self.assertEqual(set(archive.namelist()), {'main.py', 'bc_features.py', 'ppo_context.py', 'weights.json'})
            self.assertNotIn('import torch', archive.read('main.py').decode())
            weights = json.loads(archive.read('weights.json'))
            self.assertEqual(weights['feature_count'], PLACEMENT_FEATURE_COUNT)

    def test_best_baseline_iteration_zero_is_downloadable(self):
        checkpoint = deepcopy(self.checkpoint)
        checkpoint['progress'].update(episodes=0, iteration=0, best_iteration=0)
        torch.save(checkpoint, self.folder / 'best.pt')
        with patch.object(grpo_snapshots, 'run_worker', side_effect=self.fast_worker):
            result = grpo_snapshots.prepare(self.server, self.key)
        self.assertEqual(result['iteration'], 0)
        self.assertTrue(result['resumable'])

    def test_latest_has_exact_saved_progress_and_own_cache_identity(self):
        with patch.object(grpo_snapshots, 'run_worker', side_effect=self.fast_worker) as worker:
            best = grpo_snapshots.prepare(self.server, self.key)
            latest = grpo_snapshots.prepare(self.server, self.key, kind='latest')
            cached = grpo_snapshots.prepare(self.server, self.key, kind='latest')
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(latest['episodes'], 800)
        self.assertEqual(latest['iteration'], 2)
        self.assertEqual(latest['selection_metric'], 'latest_completed_iteration')
        self.assertIsNone(latest['score'])
        self.assertNotEqual(best['snapshot_id'], latest['snapshot_id'])
        self.assertEqual(latest['snapshot_id'], cached['snapshot_id'])
        self.assertEqual(grpo_snapshots.artifact(self.server, latest['snapshot_id'], 'pt').read_bytes(),
                         (self.folder / 'latest.pt').read_bytes())

    def test_archives_preserve_bc_and_grpo_formats(self):
        ppo_milestones.ensure_settings(self.folder, self.request)
        baseline = ppo_milestones.archive(self.folder, self.request,
            Path(self.request['source_checkpoint']['path']), 'baseline', 0, 0)
        milestone = ppo_milestones.archive(self.folder, self.request, self.folder / 'best.pt', 'milestone', 400, 1)
        self.assertEqual(milestone['checkpoint_format'], 'grpo')
        with patch.object(grpo_snapshots, 'run_worker', side_effect=self.fast_worker):
            original = grpo_snapshots.prepare(self.server, self.key, 'milestone', milestone_id=baseline['id'])
            trained = grpo_snapshots.prepare(self.server, self.key, 'milestone', milestone_id=milestone['id'])
        self.assertEqual(original['checkpoint_format'], 'bc')
        self.assertEqual(original['feature_count'], 189)
        self.assertFalse(original['resumable'])
        self.assertEqual(trained['checkpoint_format'], 'grpo')
        self.assertTrue(trained['resumable'])

    def test_polling_does_not_deserialize_or_write_checkpoints(self):
        before = file_hashes(self.folder)
        with patch.object(grpo_snapshots, 'stable_bytes', side_effect=AssertionError('checkpoint read')):
            state = grpo_snapshots.describe(self.server, self.key)
        self.assertTrue(state['best']['available'])
        self.assertTrue(state['latest']['available'])
        self.assertIsNone(state['best']['selection_metric'])
        self.assertEqual(file_hashes(self.folder), before)

    def test_invalid_ids_kind_checksum_and_unsafe_paths_are_rejected(self):
        for identifier in ('', '../anything', 'A' * 32):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                grpo_snapshots.prepare(self.server, identifier)
        with self.assertRaises(ValueError):
            grpo_snapshots.prepare(self.server, self.key, kind='missing')
        with self.assertRaises(ValueError):
            grpo_snapshots.prepare(self.server, self.key, milestone_id='bc-000000000')
        with patch.object(grpo_snapshots, 'run_worker', side_effect=self.fast_worker):
            result = grpo_snapshots.prepare(self.server, self.key)
        archive = grpo_snapshots.artifact(self.server, result['snapshot_id'])
        archive.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            grpo_snapshots.artifact(self.server, result['snapshot_id'])
        archive.unlink()
        archive.symlink_to(self.folder / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'inside'):
            grpo_snapshots.artifact(self.server, result['snapshot_id'])

    def test_ppo_and_corrupt_checkpoints_are_rejected_without_publishing(self):
        checkpoint = deepcopy(self.checkpoint)
        checkpoint['algorithm'] = 'ppo'
        torch.save(checkpoint, self.folder / 'best.pt')
        with patch.object(grpo_snapshots, 'run_worker', side_effect=self.fast_worker):
            with self.assertRaisesRegex(ValueError, 'not a GRPO'):
                grpo_snapshots.prepare(self.server, self.key)
            (self.folder / 'best.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                grpo_snapshots.prepare(self.server, self.key)
        self.assertEqual(list((self.data / 'grpo_exports').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
