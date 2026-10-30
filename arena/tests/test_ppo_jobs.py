"""PPO job isolation, validation, lifecycle and scaffold-export checks."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from arena import ppo, server
from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION
from arena.bc_train_worker import build_model, model_layers


class PPOJobTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_data = server.DATA
        self.previous_development = server.DEVELOPMENT
        server.DATA = Path(self.temporary.name)
        server.DEVELOPMENT = False
        server.STOP.clear()
        server.init()
        source = 'from macpoker import Bot\nclass Caller(Bot):\n    def act(self, state):\n        return state.call()\n'
        self.opponents = []
        for number in range(3):
            agent = server.register({'name': f'Caller {number}', 'filename': 'main.py',
                                     'content': base64.b64encode(source.encode()).decode()})
            self.opponents.append(agent['id'])
        actor = build_model()
        self.source = {'id': 'a' * 32, 'name': 'Completed BC', 'status': 'completed',
                       'created': server.now(), 'epoch': 3, 'epochs': 3, 'best_epoch': 2,
                       'records': 100, 'config': {}, 'history': []}
        server.save('training_jobs', self.source)
        folder = server.DATA / 'training_runs' / self.source['id']
        folder.mkdir()
        torch.save({'model': actor.state_dict(), 'epoch': 2}, folder / 'best.pt')
        self.weights = {'feature_version': FEATURE_VERSION, 'feature_count': FEATURE_COUNT,
                        'schema_version': 1, 'layers': model_layers(actor)}
        (folder / 'weights.json').write_text(json.dumps(self.weights))

    def tearDown(self):
        server.DATA = self.previous_data
        server.DEVELOPMENT = self.previous_development
        server.STOP.clear()
        self.temporary.cleanup()

    def body(self, **changes):
        settings = {'name': 'PPO test', 'source_training_id': self.source['id'],
                    'opponents': self.opponents, 'table_sizes': [4], 'device': 'cpu',
                    'target_episodes': 4, 'rollout_episodes': 4, 'update_epochs': 1,
                    'batch_size': 16, 'eval_interval': 1, 'eval_hands': 1, 'eval_seeds': 1}
        settings.update(changes)
        return settings

    def test_queue_snapshots_checkpoint_and_opponents_without_mutating_bc(self):
        before = deepcopy(server.get('training_jobs', self.source['id']))
        job = ppo.queue(server, self.body())
        folder = server.DATA / 'ppo_runs' / job['id']
        request = json.loads((folder / 'request.json').read_text())
        checkpoint = Path(request['source_checkpoint']['path'])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), request['source_checkpoint']['sha256'])
        self.assertEqual(server.get('training_jobs', self.source['id']), before)
        original = server.DATA / 'agents' / self.opponents[0] / 'main.py'
        original.write_text('changed after PPO was queued')
        copied = Path(request['opponents'][0]['path']).read_text()
        self.assertIn('class Caller', copied)
        self.assertEqual(job['config']['gamma'], 1.0)

    def test_invalid_settings_create_no_jobs(self):
        invalid = [{'name': ''}, {'target_episodes': True}, {'target_episodes': 0},
                   {'rollout_episodes': 10001}, {'learning_rate': float('nan')},
                   {'table_sizes': [4, 4]}, {'table_sizes': [3]}, {'opponents': self.opponents[:2]},
                   {'opponents': self.opponents + [self.opponents[0]]}, {'device': 'unsupported'},
                   {'gamma': 0.99}, {'eval_seeds': 0}, {'seed': -1}]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ppo.queue(server, self.body(**change))
        self.assertEqual(server.all_rows('ppo_jobs'), [])
        self.assertEqual(list((server.DATA / 'ppo_runs').iterdir()), [])

    def test_running_bc_is_ineligible_and_development_uses_cpu(self):
        running = dict(self.source, status='running')
        server.save('training_jobs', running)
        self.assertFalse(ppo.checkpoint_choices(server)[0]['eligible'])
        with self.assertRaisesRegex(ValueError, 'completed BC'):
            ppo.queue(server, self.body())
        self.assertEqual(server.get('training_jobs', running['id']), running)
        server.save('training_jobs', self.source)
        server.DEVELOPMENT = True
        job = ppo.queue(server, self.body(device='auto'))
        self.assertEqual(job['config']['device'], 'cpu')
        with self.assertRaisesRegex(ValueError, 'CPU'):
            ppo.queue(server, self.body(device='mps'))

    def test_queued_pause_resume_and_startup_hold_preserve_settings(self):
        job = ppo.queue(server, self.body())
        self.assertEqual(ppo.control(server, job['id'], 'pause')['status'], 'paused')
        resumed = ppo.control(server, job['id'], 'resume')
        self.assertEqual(resumed['config'], job['config'])
        server.init(hold_queued=True)
        self.assertEqual(server.get('ppo_jobs', job['id'])['status'], 'interrupted')
        self.assertEqual(server.get('training_jobs', self.source['id'])['status'], 'completed')

    def test_terminal_status_is_not_published_until_worker_exits(self):
        job = ppo.queue(server, self.body())
        folder = server.DATA / 'ppo_runs' / job['id']

        class Process:
            returncode = 0

            def __init__(self):
                self.calls = 0

            def poll(self):
                self.calls += 1
                if self.calls == 1:
                    (folder / 'status.json').write_text(json.dumps({'status': 'paused', 'phase': 'Paused'}))
                    return None
                return 0

        process = Process()
        with mock.patch.object(ppo.subprocess, 'Popen', return_value=process), \
                mock.patch.object(server, 'kill_tree'), \
                mock.patch.object(ppo.time, 'sleep', side_effect=lambda _: self.assertEqual(
                    server.get('ppo_jobs', job['id'])['status'], 'running')):
            result = ppo.execute(server, job)
        self.assertEqual(result['status'], 'paused')

    def test_cpu_worker_and_export_run_with_official_engine(self):
        job = ppo.queue(server, self.body())
        result = ppo.execute(server, job)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['episodes'], 4)
        self.assertEqual(result['device'], 'cpu')
        self.assertIn('test', result)
        self.assertIn('baseline_test', result)
        agent = ppo.export_agent(server, job['id'])
        self.assertEqual(ppo.export_agent(server, job['id'])['id'], agent['id'])
        exported = server.get('ppo_jobs', job['id'])
        self.assertEqual(len(exported['export_validation']), 5)
        self.assertTrue(all(set(check['verdicts']) == {'OK'} for check in exported['export_validation']))
        self.assertEqual(server.get('training_jobs', self.source['id']), self.source)


if __name__ == '__main__':
    unittest.main()
