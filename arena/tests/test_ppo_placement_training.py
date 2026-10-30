"""New placement workers retain exact checkpoints and objective metadata."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from arena import ppo_train_worker as worker
from arena.bc_features import FEATURE_VERSION, action_options, encode_observation, forward_logits
from arena.bc_train_worker import build_model
from arena.tests.test_ppo_training import observation, settings


def fake_rollout(actor, critic, opponents, config, episode_start, episode_count, work_dir, check_pause=None):
    features = encode_observation(observation()) + [0.0] * 7
    _, mask = action_options(observation())
    with torch.no_grad():
        inputs = torch.tensor([features])
        log_probabilities = worker.masked_log_probabilities(actor(inputs), torch.tensor([mask]))[0]
        value = critic(inputs).item()
    transitions = []
    for number in range(2):
        action = number + 1
        transitions.append({'features': features, 'mask': mask, 'action': action,
                            'old_log_prob': log_probabilities[action].item(), 'value': value,
                            'reward': .5 if number == 1 else 0, 'done': number == 1,
                            'episode_id': episode_start + number * 99})
    return {'transitions': transitions, 'episodes': episode_count, 'hands_by_size': {'4': episode_count},
            'chips_by_size': {'4': 20}, 'train_reward': .5, 'games': episode_count // 100}


def fake_evaluate(actor, opponents, config, work_dir, split='validation', check_pause=None):
    if check_pause:
        check_pause()
    name = Path(work_dir).name
    score, chips = {'iteration-1': (-.5, 1000), 'iteration-2': (.5, -1000),
                    'iteration-3': (.2, 2000)}.get(name, (0, 7))
    return {'bb100': chips, 'selection_score': score, 'selection_metric': 'placement_chip_score',
            'hands': 400, 'chips': chips * 8, 'split': split, 'failures': 0,
            'by_table_size': {'4': {'bb100': chips, 'selection_score': score}}}


class PlacementTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.source = self.folder / 'source.pt'
        torch.manual_seed(42)
        torch.save({'model': build_model().state_dict()}, self.source)
        worker.PAUSE_SIGNAL = False

    def tearDown(self):
        worker.PAUSE_SIGNAL = False
        self.temporary.cleanup()

    def request(self, name, **changes):
        folder = self.folder / name
        folder.mkdir()
        config = settings(target_episodes=300, rollout_episodes=100, gae_lambda=1.0, eval_hands=100)
        config.update(reward_mode='game_placement', placement_weight=.9, chip_scale_bb100=100)
        config.update(changes)
        request = {'job_id': name, 'feature_version': FEATURE_VERSION,
                   'source_checkpoint': {'path': str(self.source), 'sha256': worker.file_hash(self.source)},
                   'opponents': [], 'config': config}
        worker.write_json(folder / 'request.json', request)
        return request, folder

    def train(self, request, folder):
        with patch('arena.ppo_rollout.collect_rollout', side_effect=fake_rollout):
            with patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate):
                worker.train(request, folder, {})

    def test_best_uses_round_objective_not_chips_and_exports_context(self):
        request, folder = self.request('placement')
        source_bytes = self.source.read_bytes()
        self.train(request, folder)
        result = json.loads((folder / 'result.json').read_text())
        self.assertEqual(result['best_iteration'], 2)
        self.assertEqual(result['best_score'], .5)
        self.assertEqual(result['selection_metric'], 'placement_chip_score')
        self.assertEqual(result['best_validation']['bb100'], -1000)
        self.assertEqual(result['latest_validation']['bb100'], 2000)
        self.assertEqual(result['history'][0]['train_reward'], .5)
        self.assertEqual(result['history'][1]['validation_score'], .5)
        latest = torch.load(folder / 'latest.pt', weights_only=False)
        self.assertEqual(latest['feature_count'], 196)
        self.assertEqual(latest['context_version'], 1)
        self.assertEqual(latest['reward_mode'], 'game_placement')
        self.assertEqual(latest['reward_config']['gae_lambda'], 1)
        weights = json.loads((folder / 'weights.json').read_text())
        self.assertEqual(weights['feature_count'], 196)
        self.assertEqual(weights['context_version'], 1)
        self.assertEqual(weights['selection_metric'], 'placement_chip_score')
        from arena.ppo_models import build_actor
        best = torch.load(folder / 'best.pt', weights_only=False)
        actor = build_actor(best['actor'], context=True)
        features = encode_observation(observation()) + [.2] * 7
        expected = actor(torch.tensor(features)).detach().tolist()
        actual = forward_logits(features, weights['layers'])
        for left, right in zip(expected, actual):
            self.assertAlmostEqual(left, right, places=6)
        self.assertEqual(source_bytes, self.source.read_bytes())

    def test_pause_discards_partial_update_and_restores_identical_training(self):
        expected_request, expected_folder = self.request('reference')
        self.train(expected_request, expected_folder)
        request, folder = self.request('resumed')
        original_update = worker.update_policy
        calls = 0

        def interrupted_update(*arguments, **keywords):
            nonlocal calls
            calls += 1
            result = original_update(*arguments, **keywords)
            if calls == 2:
                raise worker.TrainingPaused()
            return result

        with patch.object(worker, 'update_policy', side_effect=interrupted_update):
            with self.assertRaises(worker.TrainingPaused):
                self.train(request, folder)
        saved = torch.load(folder / 'latest.pt', weights_only=False)
        self.assertEqual(saved['progress']['episodes'], 100)
        self.train(request, folder)
        expected = torch.load(expected_folder / 'latest.pt', weights_only=False)
        actual = torch.load(folder / 'latest.pt', weights_only=False)
        self.assertEqual(actual['progress'], expected['progress'])
        self.assertTrue(torch.equal(actual['torch_rng'], expected['torch_rng']))
        for component in ('actor', 'critic'):
            for name in actual[component]:
                self.assertTrue(torch.equal(actual[component][name], expected[component][name]))

    def test_invalid_game_horizons_and_metadata_mismatches_are_rejected(self):
        for number, change in enumerate(({'gae_lambda': .95}, {'target_episodes': 301},
                                         {'rollout_episodes': 101}, {'eval_hands': 10})):
            request, folder = self.request(str(number), **change)
            with self.assertRaises(ValueError):
                self.train(request, folder)
        request, folder = self.request('metadata')
        self.train(request, folder)
        saved = torch.load(folder / 'latest.pt', weights_only=False)
        saved['context_version'] = 7
        torch.save(saved, folder / 'latest.pt')
        with self.assertRaisesRegex(ValueError, 'reward or context'):
            self.train(request, folder)

    def test_context_checkpoint_archives_at_exact_50000_and_final_game_boundary(self):
        from arena import ppo_milestones
        request, folder = self.request('archives', target_episodes=50100, rollout_episodes=3000)
        ppo_milestones.ensure_settings(folder, request)
        # Exercise real archive/checkpoint code with tiny synthetic rollouts;
        # running another benchmark is unnecessary for this boundary check.
        with patch.object(ppo_milestones, 'evaluate_snapshot', return_value={}):
            self.train(request, folder)
        milestone = ppo_milestones.snapshot(folder, 'episode-000050000')
        final = ppo_milestones.snapshot(folder, 'final-000050100')
        self.assertEqual(milestone['feature_count'], 196)
        self.assertEqual(final['feature_count'], 196)
        checkpoint = torch.load(milestone['checkpoint_path'], weights_only=False)
        self.assertEqual(checkpoint['progress']['episodes'], 50000)
        self.assertEqual(checkpoint['reward_mode'], 'game_placement')
        episodes = [row['episodes'] for row in checkpoint['progress']['history']]
        self.assertEqual(episodes[-2:], [48000, 50000])
        final_checkpoint = torch.load(final['checkpoint_path'], weights_only=False)
        self.assertEqual(final_checkpoint['progress']['episodes'], 50100)


if __name__ == '__main__':
    unittest.main()
