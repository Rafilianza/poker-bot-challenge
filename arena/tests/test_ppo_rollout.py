"""Small CPU-only checks against the real poker engine and subprocess bots."""

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from arena import ppo_rollout
from arena.bc_features import FEATURE_COUNT


class PauseRequested(Exception):
    pass


class PpoRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ppo-rollout-test-")
        self.folder = Path(self.temporary.name)
        self.work_dir = self.folder / "work"
        self.actor = torch.nn.Linear(FEATURE_COUNT, 7)
        self.critic = torch.nn.Linear(FEATURE_COUNT, 1)
        with torch.no_grad():
            self.actor.weight.zero_()
            self.actor.bias.copy_(torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
            self.critic.weight.zero_()
            self.critic.bias.fill_(0.25)
        self.config = {"table_sizes": [4], "game_hands": 2, "seed": 37,
                       "eval_hands": 1, "eval_seeds": 1}
        self.opponents = [self.make_opponent("Caller", "return state.call()")]

    def tearDown(self):
        self.temporary.cleanup()

    def make_opponent(self, name, action):
        folder = self.folder / name
        folder.mkdir()
        lines = ["from pathlib import Path", "from macpoker import Bot", "class Opponent(Bot):",
                 "    def act(self, state):"]
        lines.extend("        " + line for line in action.splitlines())
        (folder / "main.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"id": name, "name": name, "hash": "test-snapshot", "path": str(folder)}

    def prefer_fold(self):
        with torch.no_grad():
            self.actor.bias.fill_(-40.0)
            self.actor.bias[0] = 40.0
            self.actor.bias[1] = 0.0

    def collect(self, episode_count=4, episode_start=0, config=None, check_pause=None):
        return ppo_rollout.collect_rollout(
            self.actor, self.critic, self.opponents, config or self.config,
            episode_start, episode_count, self.work_dir, check_pause,
        )

    def test_rollout_is_deterministic_exact_and_does_not_change_training_rng_or_models(self):
        self.actor.train()
        self.critic.train()
        parameters = deepcopy(self.actor.state_dict())
        rng_before = torch.get_rng_state().clone()
        config = {**self.config, "table_sizes": [4, 5, 6]}
        first = self.collect(7, config=config)
        second = self.collect(7, config=config)
        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], 7)
        self.assertEqual(first["hands_by_size"], {"4": 3, "5": 2, "6": 2})
        self.assertEqual([hand["episode_id"] for hand in first["hand_summaries"]], list(range(7)))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(self.actor.training)
        self.assertTrue(self.critic.training)
        for name, value in parameters.items():
            self.assertTrue(torch.equal(value, self.actor.state_dict()[name]))
        self.assertEqual(list(self.work_dir.iterdir()), [])

    def test_actions_have_legal_masks_correct_log_probabilities_and_terminal_rewards(self):
        rollout = self.collect(5)
        self.assertTrue(rollout["transitions"])
        transitions_by_hand = {}
        for transition in rollout["transitions"]:
            self.assertEqual(len(transition["features"]), FEATURE_COUNT)
            self.assertEqual(len(transition["mask"]), 7)
            action = transition["action"]
            self.assertTrue(transition["mask"][action])
            logits = self.actor.bias.detach().clone()
            logits[~torch.tensor(transition["mask"])] = -torch.inf
            expected_log_probability = torch.log_softmax(logits, dim=0)[action].item()
            self.assertAlmostEqual(transition["old_log_prob"], expected_log_probability, places=6)
            self.assertAlmostEqual(transition["value"], 0.25)
            self.assertTrue(math.isfinite(transition["old_log_prob"]))
            transitions_by_hand.setdefault(transition["episode_id"], []).append(transition)
        for hand in rollout["hand_summaries"]:
            transitions = transitions_by_hand.get(hand["episode_id"], [])
            self.assertEqual(len(transitions), hand["decisions"])
            self.assertEqual(hand["reward_bb"], hand["chips"] / 2.0)
            if transitions:
                self.assertTrue(transitions[-1]["done"])
                self.assertEqual(transitions[-1]["reward"], hand["reward_bb"])
                self.assertTrue(all(not row["done"] and row["reward"] == 0.0 for row in transitions[:-1]))
        self.assertEqual(sum(rollout["chips_by_size"].values()),
                         sum(hand["chips"] for hand in rollout["hand_summaries"]))

    def test_encoder_receives_only_the_learner_observation_before_the_action(self):
        received = []
        real_encode = ppo_rollout.encode_observation

        def observe(observation):
            received.append(deepcopy(observation))
            return real_encode(observation)

        with mock.patch.object(ppo_rollout, "encode_observation", side_effect=observe):
            rollout = self.collect(3)
        self.assertEqual(len(received), len(rollout["transitions"]))
        visible_board = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
        for observation, transition in zip(received, rollout["transitions"]):
            self.assertEqual(len(observation["hole"]), 2)
            self.assertEqual(len(observation["board"]), visible_board[observation["street"]])
            self.assertNotIn("holes", observation)
            self.assertNotIn("revealed", observation)
            self.assertNotIn("deltas", observation)
            self.assertEqual(transition["features"], real_encode(observation))
            self.assertEqual(sum(transition["features"][:52]), 2.0)
            self.assertEqual(sum(transition["features"][52:104]), len(observation["board"]))

    def test_small_blind_fold_reward_includes_the_posted_blind(self):
        self.prefer_fold()
        rollout = self.collect(1, episode_start=1, config={**self.config, "game_hands": 1})
        hand = rollout["hand_summaries"][0]
        self.assertEqual(hand["seat"], 1)
        self.assertEqual(hand["chips"], -1)
        self.assertEqual(hand["reward_bb"], -0.5)
        self.assertEqual(rollout["transitions"][-1]["action"], 0)
        self.assertEqual(rollout["transitions"][-1]["reward"], -0.5)

    def test_invalid_rollout_sizes_and_counts_fail_before_starting_games(self):
        for count in (0, -1, True, 10001):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    self.collect(count)
        for sizes in ([], [4, 4], [3], [{}], [True]):
            with self.subTest(sizes=sizes):
                with self.assertRaises(ValueError):
                    self.collect(1, config={**self.config, "table_sizes": sizes})
        self.assertFalse(self.work_dir.exists())

    def test_hand_without_a_learner_decision_still_counts_without_fake_transition(self):
        self.opponents = [self.make_opponent("Folder", "return state.fold()")]
        rollout = self.collect(1, episode_start=2, config={**self.config, "game_hands": 1})
        self.assertEqual(rollout["episodes"], 1)
        self.assertEqual(rollout["transitions"], [])
        hand = rollout["hand_summaries"][0]
        self.assertEqual(hand["seat"], 2)
        self.assertEqual(hand["decisions"], 0)
        self.assertEqual(hand["chips"], 1)
        self.assertEqual(hand["reward_bb"], 0.5)

    def test_opponent_writes_only_to_its_disposable_copy(self):
        opponent = self.make_opponent("Writer", 'Path("changed.txt").write_text("local copy")\nreturn state.call()')
        self.opponents = [opponent]
        source = Path(opponent["path"]) / "main.py"
        original = source.read_bytes()
        self.collect(2)
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(source.with_name("changed.txt").exists())
        self.assertEqual(list(self.work_dir.iterdir()), [])

    def test_failed_opponent_game_is_rejected_and_processes_are_closed(self):
        self.opponents = [self.make_opponent("Broken", 'raise RuntimeError("test opponent failure")')]
        processes = []
        real_start = ppo_rollout.SubprocessTransport.start

        def tracked_start(transport):
            real_start(transport)
            processes.append(transport.proc)

        with mock.patch.object(ppo_rollout.SubprocessTransport, "start", tracked_start):
            with self.assertRaisesRegex(ppo_rollout.RolloutError, "agent failed"):
                self.collect(2)
        self.assertTrue(processes)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(list(self.work_dir.iterdir()), [])

    def test_pause_exception_closes_opponents_and_removes_working_copies(self):
        processes = []
        real_start = ppo_rollout.SubprocessTransport.start

        def tracked_start(transport):
            real_start(transport)
            processes.append(transport.proc)

        def pause_after_start():
            if processes:
                raise PauseRequested()

        with mock.patch.object(ppo_rollout.SubprocessTransport, "start", tracked_start):
            with self.assertRaises(PauseRequested):
                self.collect(3, check_pause=pause_after_start)
        self.assertTrue(processes)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(list(self.work_dir.iterdir()), [])

    def test_evaluation_rotates_all_seats_and_uses_disjoint_test_seeds(self):
        config = {**self.config, "table_sizes": [4, 5, 6]}
        original_rng = torch.get_rng_state().clone()
        first = ppo_rollout.evaluate_policy(self.actor, self.opponents, config, self.work_dir)
        second = ppo_rollout.evaluate_policy(self.actor, self.opponents, config, self.work_dir)
        test = ppo_rollout.evaluate_policy(self.actor, self.opponents, config, self.work_dir, split="test")
        self.assertEqual(first, second)
        self.assertEqual(first["hands"], 4 + 5 + 6)
        self.assertEqual(first["failures"], 0)
        for size in (4, 5, 6):
            row = first["by_table_size"][str(size)]
            self.assertEqual(row["hands"], size)
            self.assertEqual(row["bb100"], row["chips"] / 2.0 / size * 100.0)
        mean_rate = sum(row["bb100"] for row in first["by_table_size"].values()) / 3
        self.assertEqual(first["bb100"], mean_rate)
        self.assertTrue(set(first["seed_groups"]).isdisjoint(test["seed_groups"]))
        self.assertIn("not a confidence interval", first["seed_group_variation"]["description"])
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(list(self.work_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
