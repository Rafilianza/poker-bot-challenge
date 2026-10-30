"""Self-play settings keep a separate, fixed external benchmark."""

import json
from pathlib import Path
import unittest

from arena import ppo, server
from arena.tests import test_ppo_jobs


class SelfPlayJobTests(unittest.TestCase):
    # Reuse the small BC checkpoint and registered opponents from PPO tests.
    setUp = test_ppo_jobs.PPOJobTests.setUp
    tearDown = test_ppo_jobs.PPOJobTests.tearDown
    body = test_ppo_jobs.PPOJobTests.body

    def test_all_modes_freeze_the_external_benchmark(self):
        jobs = []
        for mode in ('bots', 'self_play', 'mixed'):
            job = ppo.queue(server, self.body(training_mode=mode))
            jobs.append(job)
            request = json.loads((server.DATA / 'ppo_runs' / job['id'] / 'request.json').read_text())
            self.assertEqual(request['config']['training_mode'], mode)
            self.assertEqual([opponent['id'] for opponent in request['opponents']], self.opponents)
            for opponent in request['opponents']:
                self.assertIn('class Caller', Path(opponent['path']).read_text())
        original = server.DATA / 'agents' / self.opponents[0] / 'main.py'
        original.write_text('changed after queueing')
        for job in jobs:
            copied = server.DATA / 'ppo_runs' / job['id'] / 'opponents' / self.opponents[0] / 'main.py'
            self.assertIn('class Caller', copied.read_text())

    def test_old_clients_keep_existing_bot_mode(self):
        job = ppo.queue(server, self.body())
        self.assertEqual(job['config']['training_mode'], 'bots')
        self.assertNotIn('self_play_history_limit', job['config'])
        self.assertTrue(ppo.state(server)['self_play_available'])

    def test_mixed_history_settings_have_bounded_defaults(self):
        default = ppo.queue(server, self.body(training_mode='mixed'))
        self.assertEqual(default['config']['self_play_history_limit'], 5)
        self.assertEqual(default['config']['self_play_history_interval'], 5)
        custom = ppo.queue(server, self.body(training_mode='mixed',
                                            self_play_history_limit=2,
                                            self_play_history_interval=3))
        self.assertEqual(custom['config']['self_play_history_limit'], 2)
        self.assertEqual(custom['config']['self_play_history_interval'], 3)

    def test_bad_modes_and_history_settings_create_no_jobs(self):
        invalid = [
            {'training_mode': 'unknown'}, {'training_mode': True},
            {'training_mode': 'mixed', 'self_play_history_limit': 0},
            {'training_mode': 'mixed', 'self_play_history_limit': 21},
            {'training_mode': 'mixed', 'self_play_history_interval': True},
            {'training_mode': 'mixed', 'self_play_history_interval': 1001},
            {'training_mode': 'mixed', 'self_play_history_interval': 1.5},
        ]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ppo.queue(server, self.body(**settings))
        self.assertEqual(server.all_rows('ppo_jobs'), [])

    def test_pure_self_play_still_requires_fixed_benchmark_opponents(self):
        with self.assertRaisesRegex(ValueError, 'fixed benchmark opponents'):
            ppo.queue(server, self.body(training_mode='self_play', opponents=[]))
        with self.assertRaisesRegex(ValueError, 'five distinct opponents'):
            ppo.queue(server, self.body(training_mode='self_play',
                                       milestone_settings={'enabled': True}))

    def test_pause_resume_does_not_rewrite_frozen_settings(self):
        job = ppo.queue(server, self.body(training_mode='mixed'))
        request_path = server.DATA / 'ppo_runs' / job['id'] / 'request.json'
        original = request_path.read_bytes()
        ppo.control(server, job['id'], 'pause')
        resumed = ppo.control(server, job['id'], 'resume')
        self.assertEqual(resumed['config'], job['config'])
        self.assertEqual(request_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
