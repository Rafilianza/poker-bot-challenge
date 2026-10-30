"""Milestone API settings must not change old PPO training request hashes."""
import base64
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from arena import ppo, ppo_milestones, server
from arena.tests import test_ppo_jobs as fixtures


class MilestoneIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PPOJobTests()
        self.fixture.setUp()
        self.previous_live = server.LIVE_PPO_DATA
        server.LIVE_PPO_DATA = None

    def tearDown(self):
        server.LIVE_PPO_DATA = self.previous_live
        self.fixture.tearDown()

    def add_opponents(self):
        source = 'from macpoker import Bot\nclass Caller(Bot):\n    def act(self, state):\n        return state.call()\n'
        for number in range(2):
            agent = server.register({'name': f'Extra opponent {number}', 'filename': 'main.py',
                                     'content': base64.b64encode(source.encode()).decode()})
            self.fixture.opponents.append(agent['id'])

    def test_legacy_api_keeps_milestones_disabled(self):
        job = ppo.queue(server, self.fixture.body())
        folder = server.DATA / 'ppo_runs' / job['id']
        self.assertFalse(ppo_milestones.summaries(folder)['enabled'])
        request = json.loads((folder / 'request.json').read_text())
        self.assertNotIn('milestone_settings', request)
        self.assertNotIn('milestone_settings', request['config'])

    def test_explicit_settings_need_five_opponents_for_six_player_evaluation(self):
        with self.assertRaisesRegex(ValueError, 'five distinct opponents'):
            ppo.queue(server, self.fixture.body(milestone_settings={'enabled': True}))
        self.assertEqual(server.all_rows('ppo_jobs'), [])
        self.assertEqual(list((server.DATA / 'ppo_runs').iterdir()), [])
        self.add_opponents()
        job = ppo.queue(server, self.fixture.body(milestone_settings={
            'enabled': True, 'interval': 50000, 'eval_hands': 100, 'eval_seeds': 3}))
        folder = server.DATA / 'ppo_runs' / job['id']
        self.assertTrue(ppo_milestones.summaries(folder)['enabled'])
        original = (folder / 'request.json').read_bytes()
        ppo.control(server, job['id'], 'pause')
        ppo.control(server, job['id'], 'resume')
        self.assertEqual(original, (folder / 'request.json').read_bytes())

    def test_invalid_milestone_settings_leave_no_jobs(self):
        self.add_opponents()
        for settings in ({'interval': 0}, {'interval': True}, {'eval_hands': 0},
                         {'eval_seeds': 0}, {'enabled': 'yes'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ppo.queue(server, self.fixture.body(milestone_settings=settings))
        self.assertEqual(server.all_rows('ppo_jobs'), [])

    def test_a_legacy_resume_adds_settings_without_rewriting_request(self):
        self.add_opponents()
        job = ppo.queue(server, self.fixture.body())
        folder = server.DATA / 'ppo_runs' / job['id']
        (folder / 'milestone-settings.json').unlink()
        original = (folder / 'request.json').read_bytes()

        def finish_process(*args, **kwargs):
            self.assertEqual((folder / 'request.json').read_bytes(), original)
            self.assertTrue(ppo_milestones.summaries(folder)['enabled'])
            (folder / 'status.json').write_text(json.dumps({'status': 'paused'}))
            process = mock.Mock()
            process.poll.return_value = 0
            return process

        with mock.patch.object(ppo.subprocess, 'Popen', side_effect=finish_process), \
                mock.patch.object(server, 'kill_tree'):
            result = ppo.execute(server, job)
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(hashlib.sha256(original).hexdigest(), hashlib.sha256((folder / 'request.json').read_bytes()).hexdigest())

    def test_stale_queued_copy_cannot_restart_a_paused_job(self):
        job = ppo.queue(server, self.fixture.body())
        ppo.control(server, job['id'], 'pause')
        with mock.patch.object(ppo.subprocess, 'Popen') as launch:
            self.assertEqual(ppo.execute(server, job)['status'], 'paused')
        launch.assert_not_called()

    def test_snapshot_route_is_outside_work_queue_lock_and_live_is_read_only(self):
        from arena import ppo_snapshots
        handler = object.__new__(server.Handler)
        handler.server = SimpleNamespace(server_port=8768)
        job_id = 'a' * 32
        snapshot_id = 'b' * 64
        body = json.dumps({'kind': 'best'}).encode()
        handler.headers = {'Host': '127.0.0.1:8768', 'Content-Type': 'application/json',
                           'Content-Length': str(len(body))}
        handler.path = f'/api/ppo/live/{job_id}/snapshot'
        handler.rfile = io.BytesIO(body)
        handler.send = mock.Mock()

        def prepare(*args, **kwargs):
            self.assertFalse(server.LOCK._is_owned())
            self.assertTrue(kwargs['live'])
            return {'snapshot_id': snapshot_id, 'zip_available': True, 'checkpoint_available': True}

        with mock.patch.object(ppo_snapshots, 'prepare', side_effect=prepare), \
                mock.patch.object(ppo, 'control') as control:
            handler.do_POST()
        control.assert_not_called()
        status, result = handler.send.call_args.args
        self.assertEqual(status, 200)
        self.assertIn(snapshot_id, result['zip_url'])
        handler.path = f'/api/ppo/live/{job_id}/pause'
        handler.rfile = io.BytesIO(body)
        with mock.patch.object(ppo, 'control') as control:
            handler.do_POST()
        control.assert_not_called()
        self.assertEqual(handler.send.call_args.args[0], 404)


if __name__ == '__main__':
    unittest.main()
