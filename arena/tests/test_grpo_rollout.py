"""Matched environment groups must differ only in learner action sampling."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from arena import grpo_rollout
from arena.bc_features import FEATURE_COUNT
from arena.ppo_rollout import LearnerTransport
from arena.tests.test_ppo_training import observation


class GRPORolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.actor = torch.nn.Linear(FEATURE_COUNT, 7)
        self.config = {'reward_mode': 'game_placement', 'group_size': 2,
                       'table_sizes': [4, 5, 6], 'seed': 12}
        self.opponents = [{'id': 'caller', 'name': 'Caller', 'path': '/unused'}]

    def test_relative_rewards_standardize_within_group_and_ties_are_zero(self):
        advantages, spread = grpo_rollout.group_advantages([-1, 1])
        self.assertEqual(advantages, [-1, 1])
        self.assertEqual(spread, 1)
        advantages, spread = grpo_rollout.group_advantages([0.5] * 4)
        self.assertEqual(advantages, [0.0] * 4)
        self.assertEqual(spread, 0)
        with self.assertRaises(ValueError):
            grpo_rollout.group_advantages([0, math.inf])

    def test_group_shares_environment_with_independent_sampling_and_frozen_actor(self):
        calls = []
        parameters = deepcopy(self.actor.state_dict())
        rng = torch.get_rng_state().clone()

        def game(actor, critic, lineup, size, hands, seed, offset, start, folder,
                 check_pause=None, reward_config=None, learner_sample_seed=None):
            self.assertFalse(actor.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in actor.parameters()))
            self.assertIsNone(critic)
            calls.append((seed, offset, size, lineup, learner_sample_seed))
            reward = -1 if learner_sample_seed.endswith(':0') else 1
            return {'episodes': 100, 'chips': reward * 20, 'failures': 0,
                    'transitions': [{'action': 1, 'value': 0}],
                    'hand_summaries': [{'episode_id': start + number} for number in range(100)],
                    'objective': {'reward': reward}}

        with patch.object(grpo_rollout, 'play_game', side_effect=game):
            result = grpo_rollout.collect_rollout(self.actor, self.opponents, self.config, 0, 600, Path('/unused'))
        self.assertEqual(result['episodes'], 600)
        self.assertEqual(result['groups'], 3)
        self.assertEqual(result['hands_by_size'], {'4': 200, '5': 200, '6': 200})
        self.assertEqual(result['reward_std'], 1)
        self.assertEqual(result['zero_spread_fraction'], 0)
        for index in range(0, 6, 2):
            self.assertEqual(calls[index][:4], calls[index + 1][:4])
            self.assertNotEqual(calls[index][4], calls[index + 1][4])
        self.assertEqual([row['advantage'] for row in result['transitions']], [-1, 1] * 3)
        self.assertEqual(len({row['trajectory_id'] for row in result['transitions']}), 6)
        self.assertTrue(all('value' not in row for row in result['transitions']))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name in parameters:
            self.assertTrue(torch.equal(parameters[name], self.actor.state_dict()[name]))

    def test_separate_learner_rng_changes_actions_without_changing_observation(self):
        with torch.no_grad():
            self.actor.weight.zero_()
            self.actor.bias.zero_()
        first = LearnerTransport(self.actor, None, 'sample-a', 0, 4)
        replay = LearnerTransport(self.actor, None, 'sample-a', 0, 4)
        second = LearnerTransport(self.actor, None, 'sample-b', 0, 4)
        state = {**observation(), 'hand': 0}
        for count in range(30):
            first.act(state, 1000)
            replay.act(state, 1000)
            second.act(state, 1000)
        first_actions = [row['action'] for row in first.transitions]
        self.assertEqual(first_actions, [row['action'] for row in replay.transitions])
        self.assertNotEqual(first_actions, [row['action'] for row in second.transitions])
        self.assertEqual(first.transitions[0]['features'], second.transitions[0]['features'])

    def test_incomplete_group_is_rejected_before_games_start(self):
        for start, count in ((1, 200), (0, 100), (0, 10001), (0, 0)):
            with self.subTest(start=start, count=count):
                with patch.object(grpo_rollout, 'play_game') as play:
                    with self.assertRaises(ValueError):
                        grpo_rollout.collect_rollout(self.actor, self.opponents, self.config,
                                                    start, count, Path('/unused'))
                    play.assert_not_called()

    def test_pause_during_group_returns_no_partial_training_batch(self):
        def pause():
            raise RuntimeError('pause requested')
        with patch.object(grpo_rollout, 'play_game') as play:
            with self.assertRaisesRegex(RuntimeError, 'pause requested'):
                grpo_rollout.collect_rollout(self.actor, self.opponents, self.config,
                                            0, 200, Path('/unused'), pause)
            play.assert_not_called()


if __name__ == '__main__':
    unittest.main()
