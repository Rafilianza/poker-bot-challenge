"""Exercise independent self-play opponents with the real local poker SDK."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from arena import ppo_rollout, ppo_selfplay
from arena.bc_features import FEATURE_COUNT
from arena.ppo_context import PLACEMENT_FEATURE_COUNT


class SelfPlayRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='self-play-test-')
        self.folder = Path(self.temporary.name)
        self.actor = self.make_actor(FEATURE_COUNT)
        self.critic = torch.nn.Linear(FEATURE_COUNT, 1)
        self.config = {'table_sizes': [4, 5, 6], 'game_hands': 2, 'seed': 31,
                       'training_mode': 'self_play', 'eval_hands': 1, 'eval_seeds': 1}
        bot = self.folder / 'caller'
        bot.mkdir()
        (bot / 'main.py').write_text(
            'from macpoker import Bot\nclass Caller(Bot):\n'
            '    def act(self, state):\n        return state.call()\n')
        self.opponents = [{'id': 'caller', 'name': 'Caller', 'path': str(bot)}]

    def tearDown(self):
        self.temporary.cleanup()

    def make_actor(self, features):
        actor = torch.nn.Linear(features, 7)
        with torch.no_grad():
            actor.weight.zero_()
            actor.bias.copy_(torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        return actor

    def collect(self, **changes):
        config = {**self.config, **changes}
        return ppo_rollout.collect_rollout(self.actor, self.critic, self.opponents,
                                          config, 0, 6, self.folder / 'work')

    def test_real_four_five_six_player_self_play_is_legal_repeatable_and_frozen(self):
        parameters = deepcopy(self.actor.state_dict())
        global_random_state = torch.get_rng_state().clone()
        first = self.collect()
        second = self.collect()
        self.assertEqual(first, second)
        self.assertEqual(first['hands_by_size'], {'4': 2, '5': 2, '6': 2})
        self.assertEqual(first['failures'], 0)
        self.assertEqual(first['self_play']['seat_counts'], {'current': 12, 'historical': 0, 'bots': 0})
        self.assertEqual(first['self_play']['games'], 3)
        self.assertTrue(first['transitions'])
        self.assertEqual(len(first['transitions']), sum(hand['decisions'] for hand in first['hand_summaries']))
        self.assertTrue(all(row['mask'][row['action']] for row in first['transitions']))
        self.assertTrue(torch.equal(global_random_state, torch.get_rng_state()))
        self.assertTrue(self.actor.training)
        for name, parameter in self.actor.state_dict().items():
            self.assertTrue(torch.equal(parameter, parameters[name]))
        json.dumps(first['self_play'])

    def test_each_copy_uses_its_own_observation_generator_and_public_context(self):
        instances = []
        observations = []
        base_transport = ppo_selfplay.PolicyOpponentTransport

        class ObservedTransport(base_transport):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.closed = False
                instances.append(self)

            def act(self, observation, timeout_ms):
                observations.append((self.context.player_id, deepcopy(observation)))
                return super().act(observation, timeout_ms)

            def close(self):
                self.closed = True
                super().close()

        actor = ppo_rollout.frozen_cpu_model(self.make_actor(PLACEMENT_FEATURE_COUNT))
        lineup = ppo_selfplay.training_opponents(actor, [], [], 5, 'lineup', 'self_play', 0)
        with mock.patch.object(ppo_selfplay, 'PolicyOpponentTransport', ObservedTransport):
            game = ppo_rollout.play_game(actor, None, lineup, 6, 6, 'private-context', 2,
                                         0, self.folder / 'work')
        self.assertEqual(game['verdicts'], ['OK'] * 6)
        self.assertEqual(sum(game['all_chips']), 0)
        self.assertEqual([transport.context.player_id for transport in instances], list(range(1, 6)))
        self.assertEqual(len({id(transport.context) for transport in instances}), 5)
        self.assertEqual(len({transport.generator.initial_seed() for transport in instances}), 5)
        self.assertTrue(all(transport.actor is actor for transport in instances))
        self.assertTrue(all(transport.closed for transport in instances))
        self.assertTrue(all(not hasattr(transport, 'transitions') for transport in instances))
        for transport in instances:
            self.assertEqual(transport.context.scores, game['all_chips'])
        self.assertTrue(observations)
        for player, observation in observations:
            self.assertEqual(observation['players'][observation['seat']], player)
            self.assertEqual(len(observation['hole']), 2)
            self.assertNotIn('holes', observation)
            self.assertNotIn('revealed', observation)
            self.assertNotIn('deltas', observation)

    def test_current_actor_is_copied_once_per_rollout_and_shared_read_only(self):
        frozen_models = []
        real_freeze = ppo_rollout.frozen_cpu_model

        def observed_freeze(model):
            frozen = real_freeze(model)
            frozen_models.append(frozen)
            return frozen

        with mock.patch.object(ppo_rollout, 'frozen_cpu_model', side_effect=observed_freeze):
            self.collect()
        self.assertEqual(len(frozen_models), 2)  # One actor and one critic.
        self.assertIsNot(frozen_models[0], self.actor)
        self.assertFalse(frozen_models[0].training)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen_models[0].parameters()))

    def test_mixed_lineup_is_repeatable_and_uses_the_requested_proportions(self):
        actor = ppo_rollout.frozen_cpu_model(self.actor)
        history = [{'iteration': 2, 'actor': actor}, {'iteration': 4, 'actor': actor}]
        first = ppo_selfplay.training_opponents(actor, self.opponents, history, 10000, 'mix', 'mixed', 5)
        second = ppo_selfplay.training_opponents(actor, self.opponents, history, 10000, 'mix', 'mixed', 5)
        self.assertEqual(first, second)
        counts = {'current': 0, 'historical': 0, 'bots': 0}
        for opponent in first:
            counts[opponent.get('kind', 'bots')] += 1
        self.assertTrue(4800 < counts['current'] < 5200)
        self.assertTrue(2300 < counts['historical'] < 2700)
        self.assertTrue(2300 < counts['bots'] < 2700)
        self.assertEqual({opponent['iteration'] for opponent in first if opponent.get('kind') == 'historical'}, {2, 4})

    def test_empty_history_falls_back_to_current_policy(self):
        lineup = ppo_selfplay.training_opponents(self.actor, self.opponents, [], 100, 'mix', 'mixed', 0)
        self.assertEqual({opponent.get('kind', 'bots') for opponent in lineup}, {'current', 'bots'})
        pure = ppo_selfplay.training_opponents(self.actor, [], [], 5, 'pure', 'self_play', 0)
        self.assertTrue(all(opponent['kind'] == 'current' for opponent in pure))

    def test_real_mixed_rollout_reports_json_safe_opponent_counts(self):
        history = [{'iteration': 1, 'actor': ppo_rollout.frozen_cpu_model(self.actor)}]
        config = {**self.config, 'training_mode': 'mixed', 'self_play_iteration': 2, 'game_hands': 1}
        rollout = ppo_rollout.collect_rollout(self.actor, self.critic, self.opponents, config,
                                              0, 12, self.folder / 'work', self_play_history=history)
        summary = rollout['self_play']
        self.assertEqual(summary['current_iteration'], 2)
        self.assertEqual(summary['history_iterations_used'], [1])
        self.assertEqual(sum(summary['seat_counts'].values()), 48)
        self.assertTrue(all(count > 0 for count in summary['seat_counts'].values()))
        self.assertEqual(rollout['failures'], 0)
        json.dumps(summary)

    def test_legacy_bot_mode_is_exactly_unchanged(self):
        config = {key: value for key, value in self.config.items() if key != 'training_mode'}
        original = ppo_rollout.collect_rollout(self.actor, self.critic, self.opponents,
                                               config, 0, 6, self.folder / 'work')
        explicit = self.collect(training_mode='bots')
        self.assertEqual(original, explicit)
        self.assertNotIn('self_play', original)

    def test_evaluation_always_uses_external_bots(self):
        baseline = ppo_rollout.evaluate_policy(self.actor, self.opponents,
                                                {**self.config, 'training_mode': 'bots'}, self.folder / 'work')
        with mock.patch.object(ppo_selfplay, 'PolicyOpponentTransport', side_effect=AssertionError('evaluation copy')):
            for mode in ('self_play', 'mixed'):
                evaluated = ppo_rollout.evaluate_policy(self.actor, self.opponents,
                                                         {**self.config, 'training_mode': mode}, self.folder / 'work')
                self.assertEqual(evaluated, baseline)

    def test_placement_game_uses_only_learner_transitions_and_terminal_game_reward(self):
        actor = self.make_actor(PLACEMENT_FEATURE_COUNT)
        config = {**self.config, 'reward_mode': 'game_placement', 'table_sizes': [4]}
        rollout = ppo_rollout.collect_rollout(actor, None, [], config, 0, 100, self.folder / 'work')
        self.assertEqual(rollout['episodes'], 100)
        self.assertEqual(rollout['games'], 1)
        self.assertEqual(rollout['self_play']['seat_counts']['current'], 3)
        self.assertEqual(sum(row['done'] for row in rollout['transitions']), 1)
        self.assertEqual(rollout['transitions'][-1]['reward'], rollout['train_reward'])
        self.assertTrue(all(len(row['features']) == PLACEMENT_FEATURE_COUNT for row in rollout['transitions']))
        self.assertEqual(len(rollout['transitions']), sum(hand['decisions'] for hand in rollout['hand_summaries']))

    def test_placement_evaluation_remains_a_fixed_external_benchmark(self):
        actor = self.make_actor(PLACEMENT_FEATURE_COUNT)
        config = {**self.config, 'reward_mode': 'game_placement', 'table_sizes': [4], 'eval_hands': 100}
        with mock.patch.object(ppo_selfplay, 'PolicyOpponentTransport', side_effect=AssertionError('evaluation copy')):
            baseline = ppo_rollout.evaluate_policy(actor, self.opponents, config, self.folder / 'work')
            mixed = ppo_rollout.evaluate_policy(actor, self.opponents,
                                                 {**config, 'training_mode': 'mixed'}, self.folder / 'work')
        self.assertEqual(baseline, mixed)
        self.assertEqual(baseline['failures'], 0)
        for game in baseline['games']:
            self.assertEqual(game['opponents'], ['caller'] * 3)

    def test_policy_failure_is_rejected_and_all_policy_transports_close(self):
        instances = []
        base_transport = ppo_selfplay.PolicyOpponentTransport

        class BrokenTransport(base_transport):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.closed = False
                instances.append(self)

            def act(self, observation, timeout_ms):
                raise RuntimeError('test opponent failure')

            def close(self):
                self.closed = True
                super().close()

        with mock.patch.object(ppo_selfplay, 'PolicyOpponentTransport', BrokenTransport):
            with self.assertRaisesRegex(RuntimeError, 'test opponent failure'):
                self.collect()
        self.assertTrue(instances)
        self.assertTrue(all(transport.closed for transport in instances))

    def test_pause_closes_all_policy_transports(self):
        instances = []
        base_transport = ppo_selfplay.PolicyOpponentTransport

        class Paused(Exception):
            pass

        class TrackedTransport(base_transport):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.closed = False
                instances.append(self)

            def close(self):
                self.closed = True
                super().close()

        def pause():
            if instances:
                raise Paused()

        with mock.patch.object(ppo_selfplay, 'PolicyOpponentTransport', TrackedTransport):
            with self.assertRaises(Paused):
                ppo_rollout.collect_rollout(self.actor, self.critic, [], self.config, 0, 6,
                                            self.folder / 'work', check_pause=pause)
        self.assertTrue(instances)
        self.assertTrue(all(transport.closed for transport in instances))

    def test_training_mode_and_empty_opponent_pool_validation(self):
        with self.assertRaisesRegex(ValueError, 'training mode'):
            self.collect(training_mode='unknown')
        for mode in ('bots', 'mixed'):
            with self.assertRaises(ValueError):
                ppo_rollout.collect_rollout(self.actor, self.critic, [], {**self.config, 'training_mode': mode},
                                            0, 6, self.folder / 'work')
        with self.assertRaises(ValueError):
            ppo_rollout.evaluate_policy(self.actor, [], self.config, self.folder / 'work')


if __name__ == '__main__':
    unittest.main()
