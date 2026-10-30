"""New reward requests are explicit, complete-game, and fixed on resume."""
import json
import unittest

from arena import ppo, server
from arena.tests import test_ppo_jobs as fixtures


class PlacementJobTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PPOJobTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def body(self, **changes):
        body = self.fixture.body(reward_mode='game_placement', target_episodes=200,
                                 rollout_episodes=100, gae_lambda=1, eval_hands=100)
        body.update(changes)
        return body

    def test_reward_defaults_are_explicit_and_original_bc_stays_unchanged(self):
        source = server.DATA / 'training_runs' / self.fixture.source['id'] / 'best.pt'
        original = source.read_bytes()
        job = ppo.queue(server, self.body())
        self.assertEqual(job['config']['placement_weight'], 0.9)
        self.assertEqual(job['config']['chip_scale_bb100'], 100)
        request = json.loads((server.DATA / 'ppo_runs' / job['id'] / 'request.json').read_text())
        self.assertEqual(request['config']['reward_mode'], 'game_placement')
        self.assertEqual(source.read_bytes(), original)

    def test_invalid_rewards_and_partial_games_create_no_jobs(self):
        invalid = [
            {'reward_mode': 'winner'}, {'target_episodes': 150}, {'rollout_episodes': 101},
            {'gae_lambda': 0.95}, {'eval_hands': 10}, {'placement_weight': 0.49},
            {'placement_weight': 1.1}, {'placement_weight': True},
            {'placement_weight': float('nan')}, {'chip_scale_bb100': 0},
            {'chip_scale_bb100': float('inf')},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ppo.queue(server, self.body(**change))
        self.assertEqual(server.all_rows('ppo_jobs'), [])
        self.assertEqual(list((server.DATA / 'ppo_runs').iterdir()), [])

    def test_resume_keeps_the_reward_and_request_bytes(self):
        job = ppo.queue(server, self.body(placement_weight=0.95))
        path = server.DATA / 'ppo_runs' / job['id'] / 'request.json'
        original = path.read_bytes()
        ppo.control(server, job['id'], 'pause')
        resumed = ppo.control(server, job['id'], 'resume')
        self.assertEqual(resumed['config'], job['config'])
        self.assertEqual(path.read_bytes(), original)

    def test_omitted_mode_retains_legacy_hand_training(self):
        job = ppo.queue(server, self.fixture.body())
        self.assertNotIn('reward_mode', job['config'])
        self.assertEqual(job['config']['target_episodes'], 4)
        self.assertEqual(job['config']['gae_lambda'], 0.95)
        self.assertEqual(job['config']['eval_hands'], 1)

    def test_placement_defaults_select_full_games_when_fields_are_omitted(self):
        body = self.body()
        del body['gae_lambda']
        del body['eval_hands']
        job = ppo.queue(server, body)
        self.assertEqual(job['config']['gae_lambda'], 1)
        self.assertEqual(job['config']['eval_hands'], 100)


if __name__ == '__main__':
    unittest.main()
