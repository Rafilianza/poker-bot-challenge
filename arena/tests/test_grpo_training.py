"""Critic-free policy direction, equal game weighting and exact safe resume."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from arena import grpo_train_worker as grpo
from arena import ppo_train_worker
from arena.bc_features import FEATURE_VERSION, action_options, encode_observation, forward_logits
from arena.bc_train_worker import build_model
from arena.ppo_models import build_actor
from arena.tests.test_ppo_training import observation


def settings(**changes):
    result = {'reward_mode': 'game_placement', 'group_size': 2, 'target_episodes': 600,
              'rollout_episodes': 200, 'table_sizes': [4], 'update_epochs': 2,
              'batch_size': 4, 'learning_rate': 0.001, 'clip_ratio': 0.2,
              'entropy_coef': 0.0, 'bc_coef': 0.0, 'target_kl': 0.02,
              'eval_interval': 1, 'eval_hands': 100, 'eval_seeds': 1,
              'seed': 17, 'device': 'cpu', 'placement_weight': 0.9, 'chip_scale_bb100': 100}
    result.update(changes)
    return result


def examples(actor, repeat_negative=1):
    features = encode_observation(observation())
    if actor[0].in_features == 196:
        features += [0.0] * 7
    _, mask = action_options(observation())
    with torch.no_grad():
        logs = grpo.masked_log_probabilities(actor(torch.tensor([features])), torch.tensor([mask]))[0]
    rows = []
    for trajectory, action, advantage, count in ((0, 1, 1, 1), (1, 0, -1, repeat_negative)):
        for decision in range(count):
            rows.append({'features': features, 'mask': mask, 'action': action,
                         'old_log_prob': logs[action].item(), 'advantage': advantage,
                         'trajectory_id': trajectory, 'informative_group': True})
    return rows


def fake_rollout(actor, opponents, config, episode_start, episode_count, folder, check_pause=None):
    if check_pause:
        check_pause()
    return {'transitions': examples(actor, repeat_negative=3), 'episodes': episode_count,
            'chips_by_size': {'4': 10}, 'hands_by_size': {'4': episode_count},
            'train_reward': 0.0, 'groups': episode_count // (config['group_size'] * 100),
            'reward_std': 1.0, 'zero_spread_fraction': 0.0}


def fake_evaluate(actor, opponents, config, folder, split='validation', check_pause=None):
    if check_pause:
        check_pause()
    return {'selection_score': 0.25, 'bb100': 2.0, 'hands': 400, 'chips': 16,
            'split': split, 'failures': 0, 'by_table_size': {'4': {'bb100': 2.0}},
            'seed_groups': [split + '-group']}


class GRPOMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_games_have_equal_weight_despite_decision_count(self):
        weights = grpo.trajectory_weights([{'trajectory_id': 0}] + [{'trajectory_id': 1}] * 9)
        self.assertAlmostEqual(weights[0], 0.5)
        self.assertAlmostEqual(sum(weights[1:]), 0.5)

    def test_advantage_increases_good_action_and_decreases_bad_action(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(12)
            actor = build_model()
            reference = deepcopy(actor)
            rows = examples(actor)
            before = actor(torch.tensor([rows[0]['features']])).detach().softmax(-1)[0]
            optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
            metrics = grpo.update_policy(actor, reference, optimizer, rows,
                                         settings(update_epochs=1), torch.device('cpu'), 0, Path(directory))
            after = actor(torch.tensor([rows[0]['features']])).detach().softmax(-1)[0]
            self.assertGreater(after[1], before[1])
            self.assertLess(after[0], before[0])
            self.assertEqual(metrics['updates'], 1)
            self.assertGreater(metrics['approx_kl'], 0)

    def test_duplicate_decisions_and_chunk_size_do_not_change_game_weighting(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(23)
            initial = build_model()
            results = []
            for repetitions, batch_size in ((1, 10), (9, 3)):
                actor = deepcopy(initial)
                optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
                grpo.update_policy(actor, initial, optimizer, examples(actor, repetitions),
                                   settings(update_epochs=1, batch_size=batch_size),
                                   torch.device('cpu'), 0, Path(directory))
                results.append(actor.state_dict())
            for name in results[0]:
                self.assertTrue(torch.allclose(results[0][name], results[1][name], atol=1e-7), name)

    def test_zero_spread_is_skipped_even_with_entropy_and_anchor_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            actor = build_model()
            before = deepcopy(actor.state_dict())
            optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
            rows = examples(actor)
            for row in rows:
                row.update(advantage=0.0, informative_group=False)
            result = grpo.update_policy(actor, deepcopy(actor), optimizer, rows,
                                        settings(entropy_coef=1, bc_coef=1), torch.device('cpu'), 0, Path(directory))
            self.assertTrue(result['update_skipped'])
            self.assertEqual(result['updates'], 0)
            self.assertEqual(optimizer.state, {})
            for name in before:
                self.assertTrue(torch.equal(before[name], actor.state_dict()[name]))

    def test_illegal_saved_action_fails_and_large_kl_stops_more_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            actor = build_model()
            reference = deepcopy(actor)
            rows = examples(actor)
            optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
            metrics = grpo.update_policy(actor, reference, optimizer, rows,
                                         settings(target_kl=1e-12, update_epochs=4),
                                         torch.device('cpu'), 0, Path(directory))
            self.assertTrue(metrics['early_stopped'])
            self.assertEqual(metrics['updates'], 1)
            rows[0]['mask'][rows[0]['action']] = False
            with self.assertRaisesRegex(ValueError, 'saved legal mask'):
                grpo.update_policy(actor, reference, optimizer, rows, settings(),
                                   torch.device('cpu'), 0, Path(directory))


class GRPOWorkerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.source = self.folder / 'source.pt'
        torch.manual_seed(19)
        self.initial = build_model()
        torch.save({'model': self.initial.state_dict(), 'epoch': 7}, self.source)
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        ppo_train_worker.PAUSE_SIGNAL = False

    def tearDown(self):
        ppo_train_worker.PAUSE_SIGNAL = False
        self.temporary.cleanup()

    def request(self, name, **changes):
        folder = self.folder / name
        folder.mkdir()
        request = {'job_id': name, 'algorithm': 'grpo', 'feature_version': FEATURE_VERSION,
                   'config': settings(**changes), 'opponents': [],
                   'source_checkpoint': {'path': str(self.source), 'sha256': self.source_hash,
                                         'training_id': 'source', 'name': 'BC', 'epoch': 7}}
        grpo.write_json(folder / 'request.json', request)
        return request, folder

    def test_pause_resume_restores_actor_optimizer_rng_and_complete_group_counters(self):
        with patch('arena.grpo_rollout.collect_rollout', side_effect=fake_rollout), \
                patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate):
            request, expected_folder = self.request('reference')
            grpo.train(request, expected_folder, {})
            expected = torch.load(expected_folder / 'latest.pt', weights_only=False)
            request, actual_folder = self.request('resumed')
            real_update = grpo.update_policy
            calls = 0

            def interrupted_update(*arguments, **keywords):
                nonlocal calls
                calls += 1
                result = real_update(*arguments, **keywords)
                if calls == 2:
                    raise grpo.TrainingPaused()
                return result

            with patch.object(grpo, 'update_policy', side_effect=interrupted_update):
                with self.assertRaises(grpo.TrainingPaused):
                    grpo.train(request, actual_folder, {})
            saved = torch.load(actual_folder / 'latest.pt', weights_only=False)
            self.assertEqual(saved['progress']['episodes'], 200)
            self.assertEqual(saved['progress']['groups'], 1)
            grpo.train(request, actual_folder, {})
            actual = torch.load(actual_folder / 'latest.pt', weights_only=False)
        self.assertEqual(actual['progress'], expected['progress'])
        self.assertEqual(actual['algorithm'], 'grpo')
        self.assertNotIn('critic', actual)
        self.assertNotIn('critic_optimizer', actual)
        self.assertTrue(torch.equal(actual['torch_rng'], expected['torch_rng']))
        for name in actual['actor']:
            self.assertTrue(torch.equal(actual['actor'][name], expected['actor'][name]), name)
        for key, state in expected['actor_optimizer']['state'].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, actual['actor_optimizer']['state'][key][name]))
        result = json.loads((actual_folder / 'result.json').read_text())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['best_iteration'], 0)
        self.assertNotEqual(result['test']['seed_groups'], result['baseline']['seed_groups'])
        self.assertEqual(result['test']['seed_groups'], result['baseline_test']['seed_groups'])
        # Equal validation results retain the unmodified, context-extended BC.
        weights = json.loads((actual_folder / 'weights.json').read_text())
        self.assertEqual(weights['algorithm'], 'grpo')
        self.assertEqual(weights['feature_count'], 196)
        features = encode_observation(observation())
        expected_logits = self.initial(torch.tensor(features)).detach().tolist()
        actual_logits = forward_logits(features + [0.0] * 7, weights['layers'])
        for left, right in zip(expected_logits, actual_logits):
            self.assertAlmostEqual(left, right, places=6)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_milestone_hook_archives_critic_free_checkpoint_and_bc_reference(self):
        from arena import ppo_milestones
        request, folder = self.request('milestones', target_episodes=200)
        ppo_milestones.ensure_settings(folder, request, {'interval': 200, 'eval_seeds': 1})
        with patch('arena.grpo_rollout.collect_rollout', side_effect=fake_rollout), \
                patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate), \
                patch('arena.ppo_milestones.evaluate_snapshot', return_value={}):
            grpo.train(request, folder, {})
        baseline = ppo_milestones.snapshot(folder, 'bc-000000000')
        trained = ppo_milestones.snapshot(folder, 'episode-000000200')
        self.assertEqual(baseline['checkpoint_format'], 'bc')
        self.assertEqual(trained['checkpoint_format'], 'grpo')
        checkpoint = torch.load(trained['checkpoint_path'], weights_only=False)
        self.assertNotIn('critic', checkpoint)
        self.assertEqual(checkpoint['progress']['episodes'], 200)

    def test_changed_source_or_incompatible_group_boundaries_fail(self):
        request, folder = self.request('incompatible', target_episodes=300)
        with self.assertRaisesRegex(ValueError, 'complete groups'):
            grpo.train(request, folder, {})
        request, folder = self.request('checksum')
        self.source.write_bytes(self.source.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            grpo.train(request, folder, {})


if __name__ == '__main__':
    unittest.main()
