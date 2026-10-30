"""Placement rewards use complete games while retaining strict SDK checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from arena import ppo_rollout as rollout
from arena.ppo_models import build_actor, build_critic
from arena.ppo_objectives import game_reward
from arena.ppo_train_worker import calculate_advantages
from arena.tests.test_ppo_training import observation


class PlacementRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        opponent_folder = self.folder / 'opponent'
        opponent_folder.mkdir()
        (opponent_folder / 'main.py').write_text('from macpoker import Bot\nclass Caller(Bot):\n    def act(self,state):\n        return state.call()\n')
        self.opponent = {'id': 'caller', 'name': 'Caller', 'path': str(opponent_folder)}
        self.actor = build_actor(context=True)
        self.critic = build_critic(context=True)
        with torch.no_grad():
            self.actor[-1].weight.zero_()
            self.actor[-1].bias.fill_(-30)
            self.actor[-1].bias[0] = 30
            self.actor[-1].bias[1] = 0
        self.config = {'reward_mode': 'game_placement', 'table_sizes': [4],
                       'placement_weight': .9, 'chip_scale_bb100': 100,
                       'seed': 4, 'eval_hands': 100, 'eval_seeds': 1}

    def tearDown(self):
        self.temporary.cleanup()

    def test_mc_returns_reach_every_decision_and_stop_at_game_boundaries(self):
        advantages, returns = calculate_advantages([0, 0, .8, 0, -.4], [.1, .2, .3, .1, .2],
                                                    [False, False, True, False, True], gamma=1, gae_lambda=1)
        for actual, expected in zip(returns, [.8, .8, .8, -.4, -.4]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(advantages[0], .7)
        self.assertAlmostEqual(advantages[-1], -.6)

    def test_real_full_game_has_one_terminal_reward_and_public_context(self):
        rng_before = torch.get_rng_state().clone()
        parameters = deepcopy(self.actor.state_dict())
        result = rollout.collect_rollout(self.actor, self.critic, [self.opponent], self.config,
                                         0, 100, self.folder / 'work')
        transitions = result['transitions']
        self.assertEqual(result['episodes'], 100)
        self.assertEqual(result['games'], 1)
        self.assertEqual(sum(row['done'] for row in transitions), 1)
        self.assertTrue(transitions[-1]['done'])
        self.assertTrue(all(row['reward'] == 0 for row in transitions[:-1]))
        self.assertEqual(transitions[-1]['reward'], result['train_reward'])
        self.assertTrue(all(len(row['features']) == 196 for row in transitions))
        self.assertGreater(transitions[-1]['features'][189], transitions[0]['features'][189])
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        for name, value in parameters.items():
            self.assertTrue(torch.equal(value, self.actor.state_dict()[name]))
        self.assertEqual(list((self.folder / 'work').iterdir()), [])

    def test_partial_games_are_rejected_before_collection(self):
        for start, count in ((0, 1), (1, 100), (0, 150), (50, 100)):
            with self.subTest(start=start, count=count):
                with self.assertRaisesRegex(ValueError, '100-hand'):
                    rollout.collect_rollout(self.actor, self.critic, [self.opponent], self.config,
                                            start, count, self.folder / 'work')
        self.assertFalse((self.folder / 'work').exists())

    def test_terminal_reward_reaches_last_decision_when_final_hand_has_none(self):
        class NoFinalDecisionRunner:
            def __init__(self, config, transports):
                self.learner = transports[0]

            def run(self):
                self.learner.send({'type': 'hello', 'num_players': 4, 'num_hands': 100,
                                   'stack': 200, 'player': 0})
                hands = []
                for hand_number in range(100):
                    self.learner.send({'type': 'hand_start', 'hand': hand_number})
                    if hand_number == 0:
                        state = observation()
                        state.update(hand=0, players=[0, 1, 2, 3])
                        self.learner.act(state, 10000)
                    deltas = [30, -10, -10, -10] if hand_number == 0 else [0] * 4
                    self.learner.send({'type': 'hand_end', 'hand': hand_number,
                                       'players': [0, 1, 2, 3], 'deltas': deltas})
                    hands.append({'deltas_by_bot': deltas})
                return SimpleNamespace(chips=[30, -10, -10, -10], verdicts=['OK'] * 4,
                                       hands=hands, num_hands=100)

        with patch.object(rollout, 'MatchRunner', NoFinalDecisionRunner):
            result = rollout.play_game(self.actor, self.critic, [self.opponent] * 3, 4, 100,
                                       'no-final-action', 0, 0, self.folder / 'work', reward_config=self.config)
        self.assertEqual(len(result['transitions']), 1)
        self.assertEqual(result['transitions'][0]['episode_id'], 0)
        self.assertTrue(result['transitions'][0]['done'])
        self.assertEqual(result['transitions'][0]['reward'], game_reward([30, -10, -10, -10])['reward'])
        self.assertEqual(result['hand_summaries'][-1]['decisions'], 0)

    def test_evaluation_keeps_full_games_and_exact_round_selection(self):
        calls = []

        def fake_game(actor, critic, opponents, size, hands, seed, offset, episode_start, work_dir,
                      check_pause=None, reward_config=None):
            calls.append((size, hands, seed, offset))
            chips = [30] + [-30 / (size - 1)] * (size - 1)
            return {'episodes': hands, 'chips': 30, 'all_chips': chips, 'verdicts': ['OK'] * size,
                    'completed_hands': hands, 'completed': True}

        config = {**self.config, 'table_sizes': [4, 5, 6]}
        with patch.object(rollout, 'play_game', side_effect=fake_game):
            report = rollout.evaluate_policy(self.actor, [self.opponent], config, self.folder / 'eval')
        self.assertEqual(len(calls), 15)
        self.assertTrue(all(call[1] == 100 for call in calls))
        self.assertEqual(report['hands'], 1500)
        self.assertEqual(report['completed_rounds'], 3)
        self.assertEqual(report['selection_metric'], 'placement_chip_score')
        self.assertEqual(report['normalized_round_placement'], 1)
        self.assertGreater(report['selection_score'], .9)
        for size in (4, 5, 6):
            selected = [call for call in calls if call[0] == size]
            self.assertEqual(len({call[2] for call in selected}), 1)
            self.assertEqual([call[3] for call in selected], list(range(size)))
        with self.assertRaisesRegex(ValueError, '100-hand'):
            rollout.evaluate_policy(self.actor, [self.opponent], {**config, 'eval_hands': 10}, self.folder / 'eval')

    def test_tie_aware_reward_stays_bounded_and_chip_scale_does_not_dominate(self):
        tied = game_reward([30, 30, -30, -30])
        self.assertEqual(tied['points'], [3.5, 3.5, 1.5, 1.5])
        self.assertAlmostEqual(tied['placement'], 2 / 3)
        self.assertEqual(game_reward([0, 0, 0, 0])['reward'], 0)
        self.assertLessEqual(game_reward([1e9, -1e9, 0, 0])['reward'], 1)

    def test_placement_training_still_rejects_sdk_opponent_failures(self):
        path = Path(self.opponent['path']) / 'main.py'
        path.write_text('from macpoker import Bot\nclass Broken(Bot):\n    def act(self,state):\n        raise RuntimeError("broken")\n')
        with self.assertRaisesRegex(rollout.RolloutError, 'agent failed'):
            rollout.collect_rollout(self.actor, self.critic, [self.opponent], self.config,
                                    0, 100, self.folder / 'work')
        self.assertEqual(list((self.folder / 'work').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
