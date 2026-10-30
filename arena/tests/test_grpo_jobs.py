"""GRPO queue isolation, validation and shared dashboard lifecycle."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from arena import grpo, server
from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION
from arena.bc_train_worker import build_model, model_layers


class GRPOJobTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_data = server.DATA
        self.previous_development = server.DEVELOPMENT
        server.DATA = Path(self.temporary.name).resolve()
        server.DEVELOPMENT = False
        server.STOP.clear()
        server.init()
        source = 'from macpoker import Bot\nclass Caller(Bot):\n    def act(self, state):\n        return state.call()\n'
        self.opponents = []
        for number in range(5):
            agent = server.register({'name': f'Caller {number}', 'filename': 'main.py',
                                     'content': base64.b64encode(source.encode()).decode()})
            self.opponents.append(agent['id'])
        self.source = {'id': 'a' * 32, 'name': 'Completed BC', 'status': 'completed',
                       'created': server.now(), 'epoch': 3, 'epochs': 3, 'best_epoch': 2,
                       'records': 100, 'config': {}, 'history': []}
        server.save('training_jobs', self.source)
        self.source_folder = server.DATA / 'training_runs' / self.source['id']
        self.source_folder.mkdir()
        actor = build_model()
        torch.save({'model': actor.state_dict(), 'epoch': 2}, self.source_folder / 'best.pt')
        weights = {'feature_version': FEATURE_VERSION, 'feature_count': FEATURE_COUNT,
                   'schema_version': 1, 'layers': model_layers(actor)}
        (self.source_folder / 'weights.json').write_text(json.dumps(weights))

    def tearDown(self):
        server.DATA = self.previous_data
        server.DEVELOPMENT = self.previous_development
        server.STOP.clear()
        self.temporary.cleanup()

    def body(self, **changes):
        settings = {'name': 'GRPO test', 'source_training_id': self.source['id'],
                    'opponents': self.opponents, 'table_sizes': [4, 5, 6], 'device': 'cpu'}
        settings.update(changes)
        return settings

    def test_queue_freezes_sources_in_separate_storage_without_changing_bc_or_ppo(self):
        source_before = (self.source_folder / 'best.pt').read_bytes()
        ppo_job = {'id': 'b' * 32, 'name': 'Existing PPO', 'status': 'running', 'created': server.now()}
        server.save('ppo_jobs', ppo_job)
        job = grpo.queue(server, self.body())
        folder = server.DATA / 'grpo_runs' / job['id']
        request = json.loads((folder / 'request.json').read_text())
        self.assertEqual(Path(request['source_checkpoint']['path']).read_bytes(), source_before)
        self.assertEqual(hashlib.sha256(source_before).hexdigest(), request['source_checkpoint']['sha256'])
        self.assertEqual(server.get('training_jobs', self.source['id']), self.source)
        self.assertEqual(server.get('ppo_jobs', ppo_job['id']), ppo_job)
        self.assertFalse((server.DATA / 'ppo_runs' / job['id']).exists())
        original = server.DATA / 'agents' / self.opponents[0] / 'main.py'
        original.write_text('changed later')
        self.assertIn('class Caller', Path(request['opponents'][0]['path']).read_text())
        self.assertEqual(job['config']['group_size'], 4)
        self.assertEqual(job['config']['learning_rate'], 0.00003)
        self.assertEqual(job['config']['placement_weight'], 0.9)
        self.assertNotIn('critic_learning_rate', job['config'])
        self.assertNotIn('gae_lambda', job['config'])
        self.assertTrue(job['milestone_settings']['enabled'])

    def test_rejects_invalid_and_partial_groups_without_creating_jobs(self):
        invalid = [{'name': ''}, {'target_episodes': True}, {'group_size': 1}, {'group_size': 9},
                   {'target_episodes': 100}, {'rollout_episodes': 1000}, {'rollout_episodes': 10001},
                   {'learning_rate': float('nan')}, {'table_sizes': [4, 4]}, {'table_sizes': [3]},
                   {'opponents': self.opponents[:2]}, {'opponents': self.opponents + [self.opponents[0]]},
                   {'device': 'unsupported'}, {'eval_hands': 99}, {'eval_seeds': 0}, {'seed': -1},
                   {'reward_mode': 'chips'}, {'placement_weight': 0.4},
                   {'milestone_settings': {'interval': 50001}},
                   {'milestone_settings': {'eval_hands': 50}}]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                grpo.queue(server, self.body(**change))
        self.assertEqual(server.all_rows('grpo_jobs'), [])
        self.assertEqual(list((server.DATA / 'grpo_runs').iterdir()), [])

    def test_group_eight_needs_divisible_budget_and_milestones(self):
        with self.assertRaisesRegex(ValueError, 'multiple of 800'):
            grpo.queue(server, self.body(group_size=8, target_episodes=8000, rollout_episodes=1600))
        job = grpo.queue(server, self.body(group_size=8, target_episodes=8000,
                        rollout_episodes=1600, milestone_settings={'interval': 50400}))
        self.assertEqual(job['milestone_settings']['interval'], 50400)

    def test_milestones_require_five_opponents_but_plain_four_player_training_does_not(self):
        settings = self.body(opponents=self.opponents[:3], table_sizes=[4])
        with self.assertRaisesRegex(ValueError, 'five distinct'):
            grpo.queue(server, settings)
        settings['milestone_settings'] = {'enabled': False}
        self.assertEqual(grpo.queue(server, settings)['config']['table_sizes'], [4])

    def test_running_and_incompatible_bc_rejected_development_cpu_only(self):
        server.save('training_jobs', dict(self.source, status='running'))
        self.assertFalse(grpo.checkpoint_choices(server)[0]['eligible'])
        with self.assertRaisesRegex(ValueError, 'completed BC'):
            grpo.queue(server, self.body())
        server.save('training_jobs', self.source)
        server.DEVELOPMENT = True
        self.assertEqual(grpo.queue(server, self.body(device='auto'))['config']['device'], 'cpu')
        with self.assertRaisesRegex(ValueError, 'CPU'):
            grpo.queue(server, self.body(device='mps'))
        (self.source_folder / 'weights.json').write_text('{"feature_version":1,"feature_count":196}')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            grpo.queue(server, self.body())

    def test_pause_resume_startup_hold_and_checkpoint_folder(self):
        job = grpo.queue(server, self.body())
        folder = server.DATA / 'grpo_runs' / job['id']
        original_request = (folder / 'request.json').read_bytes()
        self.assertEqual(grpo.control(server, job['id'], 'pause')['status'], 'paused')
        self.assertEqual(grpo.control(server, job['id'], 'resume')['status'], 'queued')
        self.assertEqual((folder / 'request.json').read_bytes(), original_request)
        server.init(hold_queued=True)
        self.assertEqual(server.get('grpo_jobs', job['id'])['status'], 'interrupted')
        details = grpo.detail(server, job['id'])
        self.assertEqual(details['folders']['checkpoints'], str(folder))
        self.assertFalse(details['snapshot']['best']['available'])
        self.assertFalse(details['snapshot']['latest']['available'])
        self.assertEqual(server.get('training_jobs', self.source['id']), self.source)

    def test_running_pause_preserves_status_until_worker_commits(self):
        job = grpo.queue(server, self.body())
        job['status'] = 'running'
        server.save('grpo_jobs', job)
        result = grpo.control(server, job['id'], 'pause')
        self.assertEqual(result['status'], 'running')
        self.assertTrue(result['pause_requested'])
        control = server.DATA / 'grpo_runs' / job['id'] / 'control.json'
        self.assertEqual(json.loads(control.read_text()), {'action': 'pause'})

    def test_serial_queue_includes_grpo_in_created_order(self):
        job = grpo.queue(server, self.body())
        ppo_job = {'id': 'c' * 32, 'name': 'Later PPO', 'status': 'queued', 'created': '9999'}
        server.save('ppo_jobs', ppo_job)
        self.assertEqual(server.state()['lab_active'], 2)
        with mock.patch.object(server.STOP, 'wait', side_effect=[False, True]), \
                mock.patch.object(grpo, 'execute') as execute, \
                mock.patch.object(server.ppo, 'execute') as ppo_execute:
            server.worker()
        self.assertEqual(execute.call_args.args[1]['id'], job['id'])
        ppo_execute.assert_not_called()
        self.assertEqual(server.get('ppo_jobs', ppo_job['id'])['status'], 'queued')

    def test_terminal_status_waits_for_process_exit(self):
        job = grpo.queue(server, self.body())
        folder = server.DATA / 'grpo_runs' / job['id']

        class Process:
            def __init__(self):
                self.calls = 0

            def poll(self):
                self.calls += 1
                if self.calls == 1:
                    (folder / 'status.json').write_text('{"status":"paused","phase":"Paused"}')
                    return None
                return 0

        with mock.patch.object(grpo.subprocess, 'Popen', return_value=Process()) as start, \
                mock.patch.object(server, 'kill_tree'), \
                mock.patch.object(grpo.time, 'sleep', side_effect=lambda _: self.assertEqual(
                    server.get('grpo_jobs', job['id'])['status'], 'running')):
            result = grpo.execute(server, job)
        self.assertIn('arena.grpo_train_worker', start.call_args.args[0])
        self.assertEqual(result['status'], 'paused')

    def test_completed_export_uses_grpo_files(self):
        job = grpo.queue(server, self.body())
        with mock.patch.object(grpo.cloning, 'export_agent', return_value={'id': 'saved'}) as export:
            self.assertEqual(grpo.export_agent(server, job['id'])['id'], 'saved')
        export.assert_called_once_with(server, job['id'], table='grpo_jobs', folder_name='grpo_runs')


if __name__ == '__main__':
    unittest.main()
