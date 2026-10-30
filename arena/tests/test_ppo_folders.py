"""Finder actions use registered PPO folders without changing training state."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from arena import ppo, server


class PPOFolderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data_folder = Path(self.temporary.name) / 'arena data'
        data_patch = mock.patch.object(server, 'DATA', self.data_folder)
        data_patch.start()
        self.addCleanup(data_patch.stop)
        server.init()

        # Minimal records keep these tests independent of PyTorch and workers.
        self.agent = {'id': 'b' * 32, 'name': 'Exported PPO agent'}
        self.source_job = {
            'id': 'c' * 32, 'name': 'BC still running', 'status': 'running',
            'config': {'epochs': 10}, 'epoch': 3,
        }
        self.job = {
            'id': 'a' * 32, 'name': 'PPO still running', 'status': 'running',
            'agent_id': self.agent['id'], 'episodes': 40, 'pause_requested': False,
            'config': {'target_episodes': 100, 'source_training_id': self.source_job['id']},
            'source_checkpoint': {'path': '/untrusted/source/checkpoint.pt'},
        }
        server.save('agents', self.agent)
        server.save('training_jobs', self.source_job)
        server.save('ppo_jobs', self.job)
        self.agent_folder = self.data_folder / 'agents' / self.agent['id']
        self.checkpoint_folder = self.data_folder / 'ppo_runs' / self.job['id']
        self.source_folder = self.data_folder / 'training_runs' / self.source_job['id']
        for folder in (self.agent_folder, self.checkpoint_folder, self.source_folder):
            folder.mkdir(parents=True)
        (self.agent_folder / 'main.py').write_text('EXPORTED_AGENT = True\n')
        (self.checkpoint_folder / 'latest.pt').write_bytes(b'PPO checkpoint sentinel')
        (self.checkpoint_folder / 'control.json').write_text('{"action": "run"}')
        (self.source_folder / 'best.pt').write_bytes(b'BC checkpoint sentinel')

        # No test may launch Finder, including when validation unexpectedly fails.
        opener_patch = mock.patch.object(ppo.subprocess, 'run')
        self.open_process = opener_patch.start()
        self.addCleanup(opener_patch.stop)
        platform_patch = mock.patch.object(ppo.sys, 'platform', 'darwin')
        platform_patch.start()
        self.addCleanup(platform_patch.stop)

    def post(self, body, headers=None, path=None):
        """Exercise the real POST handler without a socket or background worker."""
        raw = json.dumps(body).encode()
        request = object.__new__(server.Handler)
        request.path = path or f'/api/ppo/{self.job["id"]}/open-folder'
        request.server = SimpleNamespace(server_port=8765)
        request.headers = {
            'Host': '127.0.0.1:8765',
            'Origin': 'http://127.0.0.1:8765',
            'Content-Type': 'application/json',
            'Content-Length': str(len(raw)),
        }
        request.headers.update(headers or {})
        request.rfile = io.BytesIO(raw)
        request.send = mock.Mock()
        request.do_POST()
        request.send.assert_called_once()
        return request.send.call_args.args

    def test_folder_paths_are_absolute_and_derived_from_job_identifiers(self):
        job = deepcopy(self.job)
        job['folders'] = {'agent': '/untrusted/agent', 'checkpoints': '/untrusted/checkpoints'}
        before = deepcopy(job)

        self.assertEqual(ppo.job_folders(server, job), {
            'agent': str(self.agent_folder.resolve()),
            'checkpoints': str(self.checkpoint_folder.resolve()),
        })
        self.assertEqual(job, before)

    def test_unexported_job_has_no_agent_folder_and_cannot_open_one(self):
        self.job.pop('agent_id')
        server.save('ppo_jobs', self.job)
        self.assertIsNone(ppo.job_folders(server, self.job)['agent'])
        with self.assertRaises(ValueError):
            ppo.open_folder(server, self.job['id'], 'agent')
        self.open_process.assert_not_called()

    def test_open_uses_exact_finder_arguments_for_each_known_folder(self):
        for kind, folder in (('agent', self.agent_folder), ('checkpoints', self.checkpoint_folder)):
            with self.subTest(folder=kind):
                self.open_process.reset_mock()
                result = ppo.open_folder(server, self.job['id'], kind)
                self.assertEqual(result, {'path': str(folder.resolve()), 'folder': kind})
                self.open_process.assert_called_once_with(
                    ['/usr/bin/open', str(folder.resolve())],
                    check=True, capture_output=True, text=True, timeout=5,
                )

    def test_state_and_opening_leave_jobs_checkpoint_files_and_database_unchanged(self):
        files_before = {}
        for path in self.data_folder.rglob('*'):
            if path.is_file():
                files_before[path] = path.read_bytes()

        with mock.patch.object(server, 'save') as save:
            state = ppo.state(server)
            ppo.open_folder(server, self.job['id'], 'checkpoints')
            ppo.open_folder(server, self.job['id'], 'agent')

        save.assert_not_called()
        self.assertTrue(state['folder_actions_available'])
        self.assertTrue(state['finder_available'])
        self.assertEqual(state['ppo_jobs'][0]['folders'], ppo.job_folders(server, self.job))
        self.assertEqual(server.get('ppo_jobs', self.job['id']), self.job)
        self.assertEqual(server.get('training_jobs', self.source_job['id']), self.source_job)
        files_after = {}
        for path in self.data_folder.rglob('*'):
            if path.is_file():
                files_after[path] = path.read_bytes()
        self.assertEqual(files_after, files_before)

    def test_other_platforms_keep_paths_visible_but_do_not_launch_opener(self):
        for platform in ('linux', 'win32'):
            with self.subTest(platform=platform), mock.patch.object(ppo.sys, 'platform', platform):
                state = ppo.state(server)
                self.assertTrue(state['folder_actions_available'])
                self.assertFalse(state['finder_available'])
                self.assertEqual(state['ppo_jobs'][0]['folders'], ppo.job_folders(server, self.job))
                with self.assertRaisesRegex(ValueError, 'macOS'):
                    ppo.open_folder(server, self.job['id'], 'checkpoints')
        self.open_process.assert_not_called()

    def test_only_known_folder_kinds_are_accepted(self):
        for kind in ('/tmp', '../agents', 'source_checkpoint', 'checkpoints/..', '', None, {}):
            with self.subTest(folder=kind), self.assertRaises(ValueError):
                ppo.open_folder(server, self.job['id'], kind)
        self.open_process.assert_not_called()

    def test_existing_directory_is_not_enough_without_a_registered_ppo_job(self):
        unknown_id = 'd' * 32
        (self.data_folder / 'ppo_runs' / unknown_id).mkdir()
        for key in (unknown_id, '../outside', str(self.checkpoint_folder)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ppo.open_folder(server, key, 'checkpoints')
        self.open_process.assert_not_called()

    def test_agent_folder_requires_a_registered_agent(self):
        with server.connect() as database:
            database.execute('DELETE FROM agents WHERE id=?', (self.agent['id'],))
        self.assertTrue(self.agent_folder.is_dir())
        with self.assertRaises(ValueError):
            ppo.open_folder(server, self.job['id'], 'agent')
        self.open_process.assert_not_called()

    def test_invalid_saved_identifiers_cannot_select_other_directories(self):
        for field in ('id', 'agent_id'):
            for value in ('../outside', '/tmp', 'g' * 32):
                with self.subTest(field=field, value=value):
                    job = dict(self.job)
                    job[field] = value
                    with self.assertRaises(ValueError):
                        ppo.job_folders(server, job)
        self.open_process.assert_not_called()

    def test_missing_folders_are_not_created(self):
        for kind, folder in (('agent', self.agent_folder), ('checkpoints', self.checkpoint_folder)):
            with self.subTest(folder=kind):
                saved_folder = folder.with_name(folder.name + '-saved')
                folder.rename(saved_folder)
                try:
                    with self.assertRaises(ValueError):
                        ppo.open_folder(server, self.job['id'], kind)
                    self.assertFalse(folder.exists())
                finally:
                    saved_folder.rename(folder)
        self.open_process.assert_not_called()

    def test_symlinks_cannot_open_folders_outside_the_storage_directory(self):
        outside = Path(self.temporary.name) / 'outside'
        outside.mkdir()
        for kind, folder in (('agent', self.agent_folder), ('checkpoints', self.checkpoint_folder)):
            with self.subTest(folder=kind):
                saved_folder = folder.with_name(folder.name + '-saved')
                folder.rename(saved_folder)
                folder.symlink_to(outside, target_is_directory=True)
                try:
                    with self.assertRaises(ValueError):
                        ppo.open_folder(server, self.job['id'], kind)
                finally:
                    folder.unlink()
                    saved_folder.rename(folder)
        self.open_process.assert_not_called()

    def test_symlinked_storage_parent_cannot_escape_data_folder(self):
        storage = self.data_folder / 'ppo_runs'
        outside = Path(self.temporary.name) / 'moved ppo runs'
        storage.rename(outside)
        storage.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            ppo.open_folder(server, self.job['id'], 'checkpoints')
        self.open_process.assert_not_called()

    def test_symlink_cannot_redirect_to_another_registered_jobs_folder(self):
        other_job = dict(self.job, id='d' * 32)
        server.save('ppo_jobs', other_job)
        other_folder = self.data_folder / 'ppo_runs' / other_job['id']
        other_folder.mkdir()
        saved_folder = self.checkpoint_folder.with_name('saved-checkpoints')
        self.checkpoint_folder.rename(saved_folder)
        self.checkpoint_folder.symlink_to(other_folder, target_is_directory=True)

        with self.assertRaises(ValueError):
            ppo.open_folder(server, self.job['id'], 'checkpoints')
        self.open_process.assert_not_called()

    def test_finder_failures_become_useful_validation_errors(self):
        failures = (
            OSError('Finder unavailable'),
            subprocess.CalledProcessError(1, ['/usr/bin/open'], stderr='could not open'),
            subprocess.TimeoutExpired(['/usr/bin/open'], timeout=5),
        )
        for failure in failures:
            with self.subTest(error=type(failure).__name__):
                self.open_process.side_effect = failure
                with self.assertRaisesRegex(ValueError, 'Finder'):
                    ppo.open_folder(server, self.job['id'], 'checkpoints')
        self.assertEqual(server.get('ppo_jobs', self.job['id']), self.job)

    def test_post_route_opens_only_the_folder_selected_from_saved_job(self):
        status, body = self.post({
            'folder': 'agent', 'path': '/untrusted/path', 'agent_id': 'd' * 32,
        })
        self.assertEqual(status, 200)
        self.assertEqual(body, {'path': str(self.agent_folder.resolve()), 'folder': 'agent'})
        self.assertEqual(self.open_process.call_args.args[0], ['/usr/bin/open', str(self.agent_folder.resolve())])
        self.assertEqual(server.get('ppo_jobs', self.job['id']), self.job)

    def test_post_route_rejects_cross_origin_host_and_non_json_requests(self):
        invalid_headers = (
            {'Origin': 'https://example.com'},
            {'Origin': 'null'},
            {'Host': 'example.com:8765'},
            {'Host': '127.0.0.1:1234'},
            {'Content-Type': 'text/plain'},
        )
        for headers in invalid_headers:
            with self.subTest(headers=headers):
                status, body = self.post({'folder': 'checkpoints'}, headers=headers)
                self.assertEqual(status, 403)
                self.assertIn('error', body)
        self.open_process.assert_not_called()

    def test_post_route_returns_errors_for_invalid_or_missing_folder(self):
        for request_body in ({}, {'folder': '/tmp'}, {'folder': None}, []):
            with self.subTest(body=request_body):
                status, body = self.post(request_body)
                self.assertEqual(status, 400)
                self.assertIn('error', body)
        unknown_path = f'/api/ppo/{"d" * 32}/open-folder'
        self.assertEqual(self.post({'folder': 'checkpoints'}, path=unknown_path)[0], 400)
        malformed_path = '/api/ppo/not-a-job-id/open-folder'
        self.assertEqual(self.post({'folder': 'checkpoints'}, path=malformed_path)[0], 404)
        self.open_process.assert_not_called()

    def test_post_route_reports_finder_failure_without_changing_job(self):
        self.open_process.side_effect = OSError('Finder unavailable')
        status, body = self.post({'folder': 'checkpoints'})
        self.assertEqual(status, 400)
        self.assertIn('Finder', body['error'])
        self.assertEqual(server.get('ppo_jobs', self.job['id']), self.job)


if __name__ == '__main__':
    unittest.main()
