"""Self-play versions stay frozen, bounded and exactly resumable."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from arena import ppo_train_worker as worker
from arena.bc_features import FEATURE_VERSION
from arena.bc_train_worker import build_model
from arena.tests.test_ppo_training import fake_evaluate, fake_rollout, settings


def self_play_rollout(actor, critic, opponents, config, episode_start,
                      episode_count, work_dir, check_pause=None, self_play_history=None):
    history = self_play_history or []
    rollout = fake_rollout(actor, critic, opponents, config, episode_start,
                           episode_count, work_dir, check_pause)
    # Exercise both RNG states and the restored pool during deterministic replay.
    history_bonus = 0.0
    if history:
        chosen = random.choice(history)
        history_bonus = next(chosen['actor'].parameters()).flatten()[0].item()
    for transition in rollout['transitions']:
        transition['reward'] += random.random() + torch.rand(()).item() + history_bonus
    rollout['chips_by_size']['4'] = sum(row['reward'] * 2 for row in rollout['transitions'])
    rollout['self_play'] = {
        'mode': config['training_mode'], 'current_iteration': config['self_play_iteration'],
        'seat_counts': {'current': 2, 'historical': int(bool(history)), 'bots': 0},
        'history_iterations_used': [snapshot['iteration'] for snapshot in history],
        'games': episode_count,
    }
    return rollout


class SelfPlayWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.source = self.folder / 'source.pt'
        torch.manual_seed(12)
        torch.save({'model': build_model().state_dict(), 'epoch': 7}, self.source)
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        worker.PAUSE_SIGNAL = False

    def tearDown(self):
        worker.PAUSE_SIGNAL = False
        self.temporary.cleanup()

    def request(self, name, **changes):
        folder = self.folder / name
        folder.mkdir()
        config = settings(training_mode='mixed', self_play_history_limit=2,
                          self_play_history_interval=1, target_episodes=12)
        config.update(changes)
        request = {'job_id': name, 'feature_version': FEATURE_VERSION,
                   'source_checkpoint': {'path': str(self.source), 'sha256': self.source_hash},
                   'opponents': [{'id': 'fixed-benchmark-bot'}], 'config': config}
        worker.write_json(folder / 'request.json', request)
        return request, folder

    def run_worker(self, request, folder, rollout=self_play_rollout):
        with patch('arena.ppo_rollout.collect_rollout', side_effect=rollout):
            with patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate):
                worker.train(request, folder, {})
        return torch.load(folder / 'latest.pt', weights_only=False)

    def assert_nested_equal(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key, value in expected.items():
                self.assert_nested_equal(actual[key], value)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_value, expected_value in zip(actual, expected):
                self.assert_nested_equal(actual_value, expected_value)
        else:
            self.assertEqual(actual, expected)

    def test_partial_update_resume_preserves_actor_optimizers_history_and_rng(self):
        reference_request, reference_folder = self.request('reference', self_play_history_limit=1)
        expected = self.run_worker(reference_request, reference_folder)
        request, folder = self.request('resumed', self_play_history_limit=1)
        update = worker.update_policy
        calls = 0

        def interrupted_update(*arguments, **keywords):
            nonlocal calls
            calls += 1
            result = update(*arguments, **keywords)
            if calls == 3:
                raise worker.TrainingPaused()
            return result

        with patch.object(worker, 'update_policy', side_effect=interrupted_update):
            with self.assertRaises(worker.TrainingPaused):
                self.run_worker(request, folder)
        interrupted = torch.load(folder / 'latest.pt', weights_only=False)
        self.assertEqual(interrupted['progress']['iteration'], 2)
        self.assertEqual([entry['iteration'] for entry in interrupted['self_play_state']['history']],
                         [0, 1])
        actual = self.run_worker(request, folder)
        for key in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'self_play_state',
                    'progress', 'torch_rng', 'python_rng'):
            self.assert_nested_equal(actual[key], expected[key])

    def test_pool_keeps_bc_and_bounded_periodic_versions_only_older_than_learner(self):
        observed_pools = []

        def inspect_pool(actor, critic, opponents, config, *arguments, **keywords):
            history = keywords['self_play_history']
            current_iteration = config['self_play_iteration']
            observed_pools.append([entry['iteration'] for entry in history])
            for entry in history:
                self.assertLess(entry['iteration'], current_iteration)
                self.assertFalse(entry['actor'].training)
                for parameter in entry['actor'].parameters():
                    self.assertFalse(parameter.requires_grad)
                    self.assertEqual(parameter.device.type, 'cpu')
                self.assertIsNot(entry['actor'], actor)
            return self_play_rollout(actor, critic, opponents, config, *arguments, **keywords)

        request, folder = self.request('periodic', target_episodes=16,
                                       self_play_history_interval=2)
        original_request = deepcopy(request)
        checkpoint = self.run_worker(request, folder, inspect_pool)
        self.assertEqual(request, original_request)
        self.assertEqual(observed_pools, [[], [0], [0], [0, 2], [0, 2], [0, 2, 4], [0, 2, 4], [0, 4, 6]])
        history = checkpoint['self_play_state']['history']
        self.assertEqual([entry['iteration'] for entry in history], [0, 4, 6])
        source = torch.load(self.source, weights_only=False)
        self.assert_nested_equal(history[0]['actor'], source['model'])
        for entry in history:
            self.assertEqual(entry['sha256'], worker.actor_state_hash(entry['actor']))
        progress = checkpoint['progress']
        self.assertEqual(progress['training_mode'], 'mixed')
        self.assertEqual(progress['self_play']['history_iterations_available'], [0, 4, 6])
        json.dumps(progress)  # Reports never contain tensor payloads.

    def test_one_version_pool_advances_beyond_the_initial_bc_policy(self):
        observed_pools = []

        def inspect_pool(*arguments, **keywords):
            observed_pools.append([entry['iteration'] for entry in keywords['self_play_history']])
            return self_play_rollout(*arguments, **keywords)

        request, folder = self.request('smallest-pool', self_play_history_limit=1)
        checkpoint = self.run_worker(request, folder, inspect_pool)
        self.assertEqual(observed_pools, [[], [0], [0, 1], [0, 2], [0, 3], [0, 4]])
        self.assertEqual([entry['iteration'] for entry in checkpoint['self_play_state']['history']], [0, 5])

    def test_pure_self_play_does_not_store_historical_actor_tensors(self):
        request, folder = self.request('pure', training_mode='self_play', target_episodes=4)
        checkpoint = self.run_worker(request, folder)
        self.assertEqual(checkpoint['self_play_state'],
                         {'version': 1, 'training_mode': 'self_play', 'history': []})
        self.assertEqual(checkpoint['progress']['self_play']['history_versions_available'], 0)
        weights = json.loads((folder / 'weights.json').read_text())
        self.assertNotIn('self_play_state', weights)
        self.assertNotIn('self_play', weights)

    def test_evaluations_always_receive_the_fixed_external_opponents(self):
        request, folder = self.request('fixed-evaluation', target_episodes=4)
        evaluated_splits = []

        def evaluate(actor, opponents, config, work_dir, split='validation', check_pause=None):
            self.assertEqual(opponents, request['opponents'])
            self.assertNotIn('self_play_iteration', config)
            evaluated_splits.append(split)
            return fake_evaluate(actor, opponents, config, work_dir, split, check_pause)

        with patch('arena.ppo_rollout.collect_rollout', side_effect=self_play_rollout):
            with patch('arena.ppo_rollout.evaluate_policy', side_effect=evaluate):
                worker.train(request, folder, {})
        self.assertEqual(evaluated_splits, ['validation', 'validation', 'validation', 'test', 'test'])

    def test_resume_checks_history_checksum_and_missing_state(self):
        request, folder = self.request('corrupt', target_episodes=2)
        checkpoint = self.run_worker(request, folder)
        damaged = deepcopy(checkpoint)
        first_state = damaged['self_play_state']['history'][0]['actor']
        first_state[next(iter(first_state))].flatten()[0] += 1
        torch.save(damaged, folder / 'latest.pt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_worker(request, folder)
        checkpoint.pop('self_play_state')
        torch.save(checkpoint, folder / 'latest.pt')
        with self.assertRaisesRegex(ValueError, 'missing its self-play state'):
            self.run_worker(request, folder)

    def test_legacy_mode_preserves_checkpoint_and_rollout_contract(self):
        request, folder = self.request('legacy', target_episodes=2)
        request['config'].pop('training_mode')
        checkpoint = self.run_worker(request, folder, fake_rollout)
        self.assertNotIn('self_play_state', checkpoint)
        self.assertNotIn('self_play', checkpoint['progress'])


if __name__ == '__main__':
    unittest.main()
