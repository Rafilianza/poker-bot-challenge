"""Small CPU checks for PPO math, saved masks, export and checkpoint recovery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from arena import ppo_train_worker as ppo
from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION, action_options
from arena.bc_features import encode_observation, forward_logits
from arena.bc_train_worker import build_model


def observation():
    return {'seat': 0, 'street': 'preflop', 'hole': ['As', 'Kd'], 'board': [],
            'pot': 3, 'to_call': 2, 'min_raise_to': 4, 'max_raise_to': 200,
            'can_raise': True, 'stacks': [200, 199, 198, 200],
            'street_bets': [0, 1, 2, 0], 'folded': [False] * 4,
            'button': 0, 'history': []}


def settings(**changes):
    config = {'target_episodes': 6, 'rollout_episodes': 2, 'table_sizes': [4],
              'update_epochs': 2, 'batch_size': 4, 'learning_rate': 0.001,
              'critic_learning_rate': 0.001, 'clip_ratio': 0.2, 'entropy_coef': 0.01,
              'bc_coef': 0.02, 'target_kl': 0.02, 'gamma': 1.0, 'gae_lambda': 0.95,
              'eval_interval': 1, 'eval_hands': 1, 'eval_seeds': 1,
              'seed': 17, 'device': 'cpu'}
    config.update(changes)
    return config


def fake_rollout(actor, critic, opponents, config, episode_start, episode_count,
                 work_dir, check_pause=None):
    transitions = []
    chips = 0.0
    features = encode_observation(observation())
    _, mask = action_options(observation())
    with torch.no_grad():
        logits = actor(torch.tensor([features], dtype=torch.float32))
        log_probabilities = ppo.masked_log_probabilities(logits, torch.tensor([mask]))[0]
        probabilities = log_probabilities.exp().tolist()
        value = critic(torch.tensor([features], dtype=torch.float32)).item()
    for episode in range(episode_start, episode_start + episode_count):
        if check_pause:
            check_pause()
        generator = random.Random(config['seed'] + episode)
        action = generator.choices(list(range(7)), weights=probabilities)[0]
        reward = 2.0 if action == 1 else -1.0
        transitions.append({'features': features, 'mask': mask, 'action': action,
                            'old_log_prob': log_probabilities[action].item(), 'value': value,
                            'reward': reward, 'done': True, 'episode_id': episode})
        chips += reward * 2
    return {'transitions': transitions, 'episodes': episode_count,
            'hands_by_size': {'4': episode_count}, 'chips_by_size': {'4': chips}}


def fake_evaluate(actor, opponents, config, work_dir, split='validation', check_pause=None):
    if check_pause:
        check_pause()
    # A constant score intentionally leaves the original BC actor eligible as
    # the selected best policy, even after several PPO updates.
    return {'bb100': 3.0, 'hands': 4, 'chips': 0.24,
            'by_table_size': {'4': {'bb100': 3.0, 'hands': 4, 'chips': 0.24}},
            'split': split, 'failures': 0}


class PPOMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_advantages_do_not_cross_hand_boundaries(self):
        advantages, returns = ppo.calculate_advantages(
            [0, 4, 0, -2], [1, 2, 3, 4], [False, True, False, True],
            gamma=1, gae_lambda=1,
        )
        self.assertEqual(advantages, [3, 2, -5, -6])
        self.assertEqual(returns, [4, 4, -2, -2])
        with self.assertRaisesRegex(ValueError, 'complete hands'):
            ppo.calculate_advantages([0], [1], [False])

    def test_clipping_handles_positive_and_negative_advantages(self):
        ratios = torch.tensor([1.5, 1.5, 0.5, 0.5])
        loss, kl, fraction = ppo.clipped_policy_loss(
            ratios.log(), torch.zeros(4), torch.tensor([1.0, -1.0, 1.0, -1.0]), 0.2,
        )
        self.assertAlmostEqual(loss.item(), 0.15, places=6)
        self.assertGreaterEqual(kl.item(), 0)
        self.assertEqual(fraction.item(), 1.0)
        log_probability = torch.tensor([0.7], requires_grad=True)
        clipped, _, _ = ppo.clipped_policy_loss(log_probability, torch.zeros(1), torch.ones(1), 0.2)
        clipped.backward()
        self.assertEqual(log_probability.grad.item(), 0.0)

    def test_masks_remove_illegal_probability_and_keep_entropy_finite(self):
        logits = torch.tensor([[1000.0, -1000.0, 1000.0]])
        masks = torch.tensor([[False, True, False]])
        log_probabilities = ppo.masked_log_probabilities(logits, masks)
        self.assertEqual(log_probabilities.exp().tolist(), [[0.0, 1.0, 0.0]])
        entropy = -(log_probabilities.exp() * log_probabilities).sum()
        self.assertTrue(torch.isfinite(entropy))
        with self.assertRaises(ValueError):
            ppo.masked_log_probabilities(logits, torch.zeros_like(masks))

    def test_update_respects_saved_masks_and_stops_on_large_kl(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            torch.manual_seed(17)
            actor = build_model()
            critic = ppo.build_critic()
            reference = deepcopy(actor)
            config = settings(learning_rate=0.01, target_kl=1e-12, batch_size=2)
            rollout = fake_rollout(actor, critic, [], config, 0, 8, folder)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config['learning_rate'])
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
            metrics = ppo.update_policy(actor, critic, reference, actor_optimizer, critic_optimizer,
                                        rollout['transitions'], config, torch.device('cpu'), 0, folder)
            self.assertTrue(metrics['early_stopped'])
            self.assertGreater(metrics['updates'], 0)
            self.assertGreaterEqual(metrics['approx_kl'], 0)
            self.assertLessEqual(metrics['bc_coef'], config['bc_coef'])
            invalid = deepcopy(rollout['transitions'])
            invalid[0]['mask'][invalid[0]['action']] = False
            with self.assertRaisesRegex(ValueError, 'saved legal mask'):
                ppo.update_policy(actor, critic, reference, actor_optimizer, critic_optimizer,
                                  invalid, config, torch.device('cpu'), 0, folder)

    def test_export_uses_the_same_actor_logits_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            actor = build_model()
            ppo.export_weights(Path(directory), actor, 2)
            model = json.loads((Path(directory) / 'weights.json').read_text())
            features = encode_observation(observation())
            expected = actor(torch.tensor(features)).detach().tolist()
            actual = forward_logits(features, model['layers'])
            self.assertEqual(model['feature_version'], FEATURE_VERSION)
            for left, right in zip(expected, actual):
                self.assertAlmostEqual(left, right, places=6)


class PPOWorkerTests(unittest.TestCase):
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
        ppo.PAUSE_SIGNAL = False

    def tearDown(self):
        ppo.PAUSE_SIGNAL = False
        self.temporary.cleanup()

    def request(self, folder_name, **config_changes):
        folder = self.folder / folder_name
        folder.mkdir()
        request = {'job_id': folder_name, 'feature_version': FEATURE_VERSION,
                   'source_checkpoint': {'path': str(self.source), 'sha256': self.source_hash,
                                         'training_id': 'bc-source', 'name': 'BC source', 'epoch': 7},
                   'opponents': [], 'config': settings(**config_changes)}
        ppo.write_json(folder / 'request.json', request)
        return request, folder

    def test_checkpoint_resume_replays_partial_iteration_identically(self):
        with patch('arena.ppo_rollout.collect_rollout', side_effect=fake_rollout):
            with patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate):
                reference_request, reference_folder = self.request('reference')
                ppo.train(reference_request, reference_folder, {})
                expected = torch.load(reference_folder / 'latest.pt', weights_only=False)
                request, folder = self.request('resumed')
                original_update = ppo.update_policy
                calls = 0
                def interrupted_update(*arguments, **keywords):
                    nonlocal calls
                    calls += 1
                    result = original_update(*arguments, **keywords)
                    if calls == 2:
                        # Interrupt after the actor changed, before the iteration
                        # commits. Resume must discard these unsaved parameters.
                        raise ppo.TrainingPaused()
                    return result
                with patch.object(ppo, 'update_policy', side_effect=interrupted_update):
                    with self.assertRaises(ppo.TrainingPaused):
                        ppo.train(request, folder, {})
                saved = torch.load(folder / 'latest.pt', weights_only=False)
                self.assertEqual(saved['progress']['iteration'], 1)
                self.assertEqual(saved['progress']['episodes'], 2)
                ppo.train(request, folder, {})
                actual = torch.load(folder / 'latest.pt', weights_only=False)
                self.assertEqual(actual['progress'], expected['progress'])
                for name in expected['actor']:
                    self.assertTrue(torch.equal(actual['actor'][name], expected['actor'][name]))
                for name in expected['critic']:
                    self.assertTrue(torch.equal(actual['critic'][name], expected['critic'][name]))
                result = json.loads((folder / 'result.json').read_text())
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['best_iteration'], 0)
                self.assertEqual(result['test']['split'], 'test')
                self.assertEqual(result['baseline_test']['split'], 'test')

    def test_hands_without_learner_decisions_are_counted_without_fake_updates(self):
        def no_decisions(actor, critic, opponents, config, start, count, work_dir, check_pause=None):
            return {'transitions': [], 'episodes': count,
                    'hands_by_size': {'4': count}, 'chips_by_size': {'4': count * 3}}
        request, folder = self.request('no-decisions', target_episodes=2)
        with patch('arena.ppo_rollout.collect_rollout', side_effect=no_decisions):
            with patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate):
                ppo.train(request, folder, {})
        result = json.loads((folder / 'result.json').read_text())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['episodes'], 2)
        self.assertEqual(result['decisions'], 0)
        self.assertEqual(result['history'][0]['updates'], 0)
        self.assertIsNone(result['history'][0]['policy_loss'])

    def test_tiny_real_engine_worker_has_disjoint_test_and_matched_bc_comparison(self):
        opponent_folder = self.folder / 'opponent'
        opponent_folder.mkdir()
        source = 'from macpoker import Bot\nclass Caller(Bot):\n    def act(self, state):\n        return state.call()\n'
        opponent_path = opponent_folder / 'main.py'
        opponent_path.write_text(source)
        request, folder = self.request('real-engine', target_episodes=4, rollout_episodes=2)
        request['opponents'] = [{'id': 'caller', 'name': 'Caller', 'hash': 'frozen-test',
                                 'path': str(opponent_folder)}]
        ppo.train(request, folder, {})
        result = json.loads((folder / 'result.json').read_text())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['episodes'], 4)
        self.assertGreater(result['decisions'], 0)
        self.assertTrue(set(result['baseline']['seed_groups']).isdisjoint(result['test']['seed_groups']))
        self.assertEqual(result['test']['seed_groups'], result['baseline_test']['seed_groups'])
        self.assertEqual(result['test']['failures'], 0)
        self.assertEqual(result['baseline_test']['failures'], 0)
        self.assertEqual(opponent_path.read_text(), source)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)
        self.assertGreater(result['elapsed_seconds'], 0)
        self.assertEqual(result['best_validation']['bb100'], result['best_score'])

    def test_checkpoint_and_control_files_are_checked(self):
        request, folder = self.request('changed-checkpoint')
        self.source.write_bytes(self.source.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            ppo.train(request, folder, {})
        ppo.write_json(folder / 'control.json', {'action': 'pause'})
        with self.assertRaises(ppo.TrainingPaused):
            ppo.check_pause(folder)
        (folder / 'control.json').unlink()
        ppo.request_pause(None, None)
        with self.assertRaises(ppo.TrainingPaused):
            ppo.check_pause(folder)


if __name__ == '__main__':
    unittest.main()
