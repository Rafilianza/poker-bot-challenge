"""Tie-aware game rewards and exact second-stage duplicate-round scoring."""
import math
import unittest

from arena import ppo_milestones, ppo_objectives


def game(chips, offset, seed=0):
    size = len(chips)
    return {'chips': chips, 'table_size': size, 'offset': offset, 'seed_group': seed,
            'seed': f'group-{seed}-size-{size}',
            'opponents': [{'id': f'player-{player}', 'hash': 'fixed'} for player in range(1, size)],
            'verdicts': ['OK'] * size, 'completed_hands': 100, 'requested_hands': 100,
            'completed': True, 'error': None}


class PlacementObjectiveTests(unittest.TestCase):
    def test_game_reward_centers_ties_and_bounds_chip_component(self):
        result = ppo_objectives.game_reward([10, 10, -20], placement_weight=0.9)
        self.assertEqual(result['points'], [2.5, 2.5, 1])
        self.assertEqual(result['placement'], 0.5)
        self.assertAlmostEqual(result['reward'], 0.45 + 0.1 * math.tanh(0.05))
        self.assertEqual(ppo_objectives.game_reward([0, 0, 0])['reward'], 0)
        self.assertAlmostEqual(ppo_objectives.game_reward([10**9, -10**9])['reward'], 1)
        self.assertAlmostEqual(ppo_objectives.game_reward([-10**9, 10**9])['reward'], -1)
        self.assertEqual(ppo_objectives.reward_settings({})['reward_mode'], 'chips')

    def test_round_ranks_summed_points_instead_of_mean_game_rank_or_total_chips(self):
        games = [game([10, 0, -10], 0), game([10, -10, 0], 1), game([-10, 10, 0], 2)]
        result = ppo_objectives.aggregate_rounds(games, placement_weight=1)
        saved_round = result['rounds'][0]
        self.assertEqual(saved_round['game_points'], [7, 6, 5])
        self.assertEqual(saved_round['round_points'], [3, 2, 1])
        self.assertEqual(result['selection_score'], 1)
        mean_game_reward = sum(ppo_objectives.game_reward(row['chips'], placement_weight=1)['reward'] for row in games) / 3
        self.assertAlmostEqual(mean_game_reward, 1 / 3)
        self.assertNotEqual(result['selection_score'], mean_game_reward)
        # A large chip win has the same placement points as a small chip win.
        games[2]['chips'] = [-20000, 10000, 10000]
        changed = ppo_objectives.aggregate_rounds(games, placement_weight=1)
        self.assertGreater(changed['selection_score'], 0)
        self.assertLess(changed['chips'], 0)

    def test_second_stage_ties_share_round_places(self):
        games = [game([10, 0, -10], 0), game([-10, 10, 0], 1), game([0, -10, 10], 2)]
        result = ppo_objectives.aggregate_rounds(games)
        self.assertEqual(result['rounds'][0]['game_points'], [6, 6, 6])
        self.assertEqual(result['rounds'][0]['round_points'], [2, 2, 2])
        self.assertEqual(result['centered_round_placement'], 0)
        self.assertAlmostEqual(result['selection_score'], 0)

    def test_chip_bonus_averages_tanh_per_game_not_tanh_after_aggregation(self):
        games = [game([1000, -1000], 0), game([-100, 100], 1)]
        result = ppo_objectives.aggregate_rounds(games, placement_weight=0)
        expected = (math.tanh(5) + math.tanh(-0.5)) / 2
        self.assertAlmostEqual(result['selection_score'], expected)
        self.assertNotAlmostEqual(result['selection_score'], math.tanh(result['bb100'] / 100))

    def test_equal_table_size_weighting_and_per_seed_outputs(self):
        games = [game([30, -10, -10, -10], offset) for offset in range(4)]
        games += [game([-50, 10, 10, 10, 10, 10], offset) for offset in range(6)]
        result = ppo_objectives.aggregate_rounds(games, placement_weight=1, expected_seeds=[0], table_sizes=[4, 6])
        self.assertEqual(result['selection_score'], 0)
        self.assertEqual(result['by_table_size']['4']['selection_score'], 1)
        self.assertEqual(result['by_table_size']['6']['selection_score'], -1)
        self.assertEqual(result['by_seed_group'][0]['selection_score'], 0)
        self.assertEqual(result['expected_rounds'], 2)
        self.assertEqual(result['completed_rounds'], 2)

    def test_incomplete_or_failed_rounds_cannot_select_checkpoint(self):
        games = [game([10, -10], 0), game([-10, 10], 1)]
        for selection in (games[:1], games):
            expected_seeds = [0, 1] if len(selection) == 2 else [0]
            result = ppo_objectives.aggregate_rounds(selection, expected_seeds=expected_seeds, table_sizes=[2])
            self.assertFalse(result['eligible_for_best'])
            self.assertIsNone(result['selection_score'])
        games[0]['verdicts'][1] = 'RTE'
        result = ppo_objectives.aggregate_rounds(games)
        self.assertFalse(result['eligible_for_best'])
        self.assertEqual(result['failures'], 1)
        self.assertIsNone(result['selection_score'])

    def test_mismatched_duplicate_evidence_is_rejected(self):
        examples = []
        duplicate = [game([10, -10], 0), game([-10, 10], 0)]
        examples.append(duplicate)
        changed_opponent = [game([10, -10], 0), game([-10, 10], 1)]
        changed_opponent[1]['opponents'][0]['id'] = 'different'
        examples.append(changed_opponent)
        missing_opponents = [game([10, -10], 0), game([-10, 10], 1)]
        for row in missing_opponents:
            row.pop('opponents')
        examples.append(missing_opponents)
        for games in examples:
            with self.subTest(games=games), self.assertRaises(ValueError):
                ppo_objectives.aggregate_rounds(games)
        for chips in ([1, 1], [float('nan'), 0], [True, -1]):
            with self.subTest(chips=chips), self.assertRaises(ValueError):
                ppo_objectives.game_reward(chips)
        for settings in ({'placement_weight': True}, {'placement_weight': 1.1}, {'chip_scale_bb100': 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ppo_objectives.reward_settings(settings)

    def test_legacy_milestone_summaries_add_rounds_without_changing_files(self):
        from arena.tests.test_ppo_milestones import MilestoneTests
        fixture = MilestoneTests()
        fixture.setUpClass()
        fixture.setUp()
        try:
            request, folder = fixture.request()
            fixture.activate(request, folder)
            fixture.train_fake(request, folder)
            saved_files = {path: path.read_bytes() for path in (folder / 'milestones').rglob('*') if path.is_file()}
            summary = ppo_milestones.summaries(folder)
            self.assertEqual(summary['round_leaders']['placement_chip_score'], 'bc-000000000')
            self.assertEqual(set(summary['leaders']), {'bb100', 'normalized_placement'})
            self.assertIn('round_metrics', summary['baseline'])
            for path, original in saved_files.items():
                self.assertEqual(path.read_bytes(), original)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
