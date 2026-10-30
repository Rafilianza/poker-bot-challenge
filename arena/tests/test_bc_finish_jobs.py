"""Finishing BC preserves checkpoints, requested settings and queue isolation."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest import mock

import torch

from arena import cloning, ppo, server
from arena import bc_train_worker as training
from arena.tests import test_cloning as fixtures


class BCFinishJobTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = fixtures.TrainingTests()
        self.fixture.setUp()
        dataset = self.fixture.dataset()
        dataset.update(episodes=6, elapsed_seconds=1, kind='collection',
                       created=server.now(), config={'deals': 1})
        server.save('datasets', dataset)
        self.job = self.fixture.queue(epochs=10)
        self.request, self.folder = self.fixture.request(self.job)

    def tearDown(self):
        self.fixture.tearDown()

    def save_paused_checkpoint(self):
        """Train two tiny fixture epochs, then pause with eight still requested."""
        original_write = training.write_json

        def pause_after_second_epoch(path, value):
            original_write(path, value)
            if path.name == 'status.json' and value.get('epoch') == 2:
                original_write(self.folder / 'control.json', {'action': 'pause'})

        report = {'status': 'running', 'history': []}
        with mock.patch.object(training, 'write_json', side_effect=pause_after_second_epoch):
            with self.assertRaises(training.TrainingPaused):
                training.train(self.request, self.folder, report)
        self.job.update(report, status='paused')
        server.save('training_jobs', self.job)

    def test_finish_requires_a_validated_checkpoint_and_updates_api_eligibility(self):
        self.assertFalse(cloning.detail(server, self.job['id'])['can_finish'])
        with self.assertRaisesRegex(ValueError, 'validated epoch'):
            cloning.control(server, self.job['id'], 'finish')
        self.assertFalse((self.folder / 'control.json').exists())
        self.save_paused_checkpoint()
        self.assertTrue(cloning.detail(server, self.job['id'])['can_finish'])
        self.assertTrue(server.lab_state()['training_jobs'][0]['can_finish'])

    def test_running_finish_keeps_requested_settings_and_last_saved_weights(self):
        self.save_paused_checkpoint()
        self.job['status'] = 'running'
        server.save('training_jobs', self.job)
        original_settings = deepcopy(self.job['config'])
        before = (self.folder / 'latest.pt').read_bytes()
        result = cloning.control(server, self.job['id'], 'finish')
        self.assertEqual(result['status'], 'running')
        self.assertTrue(result['finish_requested'])
        self.assertEqual(result['config'], original_settings)
        self.assertEqual(result['epochs'], 10)
        self.assertEqual((self.folder / 'latest.pt').read_bytes(), before)
        self.assertEqual(json.loads((self.folder / 'control.json').read_text()), {'action': 'finish'})

    def test_pause_restart_and_resume_preserve_finish_intent(self):
        self.save_paused_checkpoint()
        requested = cloning.control(server, self.job['id'], 'finish')
        self.assertEqual(requested['status'], 'queued')
        paused = cloning.control(server, self.job['id'], 'pause')
        self.assertEqual(paused['status'], 'paused')
        self.assertTrue(paused['finish_requested'])
        resumed = cloning.control(server, self.job['id'], 'resume')
        self.assertTrue(resumed['finish_requested'])
        self.assertEqual(json.loads((self.folder / 'control.json').read_text())['action'], 'finish')
        server.init(hold_queued=True)
        interrupted = server.get('training_jobs', self.job['id'])
        self.assertEqual(interrupted['status'], 'interrupted')
        self.assertTrue(interrupted['finish_requested'])
        resumed = cloning.control(server, self.job['id'], 'resume')
        self.assertEqual(resumed['config']['epochs'], 10)
        self.assertEqual(json.loads((self.folder / 'control.json').read_text())['action'], 'finish')

    def test_stale_queue_copy_does_not_erase_a_new_finish_request(self):
        self.save_paused_checkpoint()
        stale = cloning.control(server, self.job['id'], 'resume')
        cloning.control(server, self.job['id'], 'finish')

        class FinishedProcess:
            def poll(self):
                return 0

        def start_process(*args, **kwargs):
            current = server.get('training_jobs', self.job['id'])
            self.assertTrue(current['finish_requested'])
            self.assertEqual(json.loads((self.folder / 'control.json').read_text())['action'], 'finish')
            cloning.write_json(self.folder / 'status.json', {'status': 'paused'})
            return FinishedProcess()

        with mock.patch.object(cloning.subprocess, 'Popen', side_effect=start_process), \
                mock.patch.object(server, 'kill_tree'):
            result = cloning.execute(server, stale)
        self.assertTrue(result['finish_requested'])

    def test_stale_queue_copy_cannot_start_a_now_paused_job(self):
        stale = deepcopy(self.job)
        cloning.control(server, self.job['id'], 'pause')
        with mock.patch.object(cloning.subprocess, 'Popen') as start:
            result = cloning.execute(server, stale)
        start.assert_not_called()
        self.assertEqual(result['status'], 'paused')

    def test_finish_waits_for_an_inflight_pause_to_exit(self):
        self.save_paused_checkpoint()
        self.job.update(status='running', pause_requested=False)
        server.save('training_jobs', self.job)
        cloning.control(server, self.job['id'], 'pause')
        process = mock.Mock()
        process.poll.side_effect = [None, 0]

        def start_process(*args, **kwargs):
            cloning.write_json(self.folder / 'status.json', {'status': 'paused'})
            return process

        def finish_while_child_is_exiting(seconds):
            self.assertFalse(cloning.detail(server, self.job['id'])['can_finish'])
            with self.assertRaisesRegex(ValueError, 'Wait until training is paused'):
                cloning.control(server, self.job['id'], 'finish')
            self.assertEqual(json.loads((self.folder / 'control.json').read_text())['action'], 'pause')

        with mock.patch.object(cloning.subprocess, 'Popen', side_effect=start_process), \
                mock.patch.object(server, 'kill_tree'), \
                mock.patch.object(cloning.time, 'sleep', side_effect=finish_while_child_is_exiting):
            result = cloning.execute(server, self.job)
        self.assertEqual(result['status'], 'paused')
        requested = cloning.control(server, self.job['id'], 'finish')
        self.assertEqual(requested['status'], 'queued')
        self.assertTrue(requested['finish_requested'])

    def test_shutdown_delays_finalization_without_forgetting_it(self):
        self.save_paused_checkpoint()
        job = cloning.control(server, self.job['id'], 'finish')
        server.STOP.set()
        folder = self.folder

        class PausingProcess:
            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                cloning.write_json(folder / 'status.json', {'status': 'paused'})
                return 0

        with mock.patch.object(cloning.subprocess, 'Popen', return_value=PausingProcess()), \
                mock.patch.object(server, 'kill_tree'), mock.patch.object(cloning.time, 'sleep'):
            result = cloning.execute(server, job)
        self.assertEqual(result['status'], 'paused')
        self.assertTrue(result['finish_requested'])
        self.assertEqual(json.loads((self.folder / 'control.json').read_text())['action'], 'pause')
        server.STOP.clear()
        cloning.control(server, self.job['id'], 'resume')
        self.assertEqual(json.loads((self.folder / 'control.json').read_text())['action'], 'finish')

    def test_early_finish_exports_best_model_and_makes_it_eligible_for_ppo(self):
        self.save_paused_checkpoint()
        best_before = (self.folder / 'best.pt').read_bytes()
        latest_before = (self.folder / 'latest.pt').read_bytes()
        unrelated = {'id': 'b' * 32, 'name': 'Other training', 'status': 'running',
                     'created': server.now(), 'epoch': 5, 'config': {'epochs': 20}}
        server.save('training_jobs', unrelated)
        job = cloning.control(server, self.job['id'], 'finish')
        result = cloning.execute(server, job)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['epoch'], 2)
        self.assertEqual(result['epochs'], 10)
        self.assertTrue(result['stopped_early'])
        self.assertFalse(result['finish_requested'])
        self.assertIn('test', result['metrics'])
        self.assertEqual((self.folder / 'best.pt').read_bytes(), best_before)
        self.assertEqual((self.folder / 'latest.pt').read_bytes(), latest_before)
        best = torch.load(self.folder / 'best.pt', map_location='cpu', weights_only=False)
        self.assertEqual(result['best_epoch'], best['epoch'])
        expected = training.build_model()
        expected.load_state_dict(best['model'])
        weights = json.loads((self.folder / 'weights.json').read_text())
        self.assertEqual(weights['layers'], training.model_layers(expected))
        self.assertEqual(server.get('training_jobs', unrelated['id']), unrelated)
        self.assertFalse(cloning.detail(server, job['id'])['can_finish'])
        choices = {row['id']: row for row in ppo.checkpoint_choices(server)}
        self.assertTrue(choices[job['id']]['eligible'])
        self.assertFalse(choices[unrelated['id']]['eligible'])
        agent = cloning.export_agent(server, job['id'])
        self.assertEqual(cloning.export_agent(server, job['id'])['id'], agent['id'])
        exported = server.get('training_jobs', job['id'])
        self.assertTrue(all(set(row['verdicts']) == {'OK'} for row in exported['export_validation']))
        checksum = hashlib.sha256((self.folder / 'weights.json').read_bytes()).hexdigest()
        again = cloning.control(server, job['id'], 'finish')
        self.assertEqual(again['status'], 'completed')
        self.assertEqual(hashlib.sha256((self.folder / 'weights.json').read_bytes()).hexdigest(), checksum)


if __name__ == '__main__':
    unittest.main()
