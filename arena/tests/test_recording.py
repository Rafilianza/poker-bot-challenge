"""Decision recording checks using the real engine and temporary arena data."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from arena import server


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_data = server.DATA
        server.DATA = Path(self.temporary_directory.name)
        server.CANCEL.clear()
        server.STOP.clear()
        server.init()

    def tearDown(self):
        server.DATA = self.previous_data
        server.CANCEL.clear()
        server.STOP.clear()
        self.temporary_directory.cleanup()

    def register_agent(self, name, action='return state.call()', log_observations=False):
        source = 'import json\nimport sys\nfrom macpoker import Bot\n'
        source += 'class TestBot(Bot):\n    def act(self, state):\n'
        if log_observations:
            source += '        print("OBS:" + json.dumps(state.raw()), file=sys.stderr)\n'
        for line in action.splitlines():
            source += '        ' + line + '\n'
        return server.register({
            'name': name,
            'filename': 'main.py',
            'content': base64.b64encode(source.encode()).decode(),
        })

    def queue_run(self, agents, teachers=None, deals=3, mode='tournament'):
        body = {
            'mode': mode,
            'agents': [agent['id'] for agent in agents],
            'seeds': ['recording-test'],
            'deals': deals,
        }
        if teachers is not None:
            body['record_training'] = True
            body['teachers'] = [teacher['id'] for teacher in teachers]
        return server.queue_run(body)

    def read_game(self, run, result):
        folder = server.DATA / 'runs' / run['id']
        history = json.loads((folder / result['history_file']).read_text())
        records = []
        if 'training' in result:
            training_path = folder / result['training']['file']
            records = [json.loads(line) for line in training_path.read_text().splitlines()]
        return history, records

    def test_queue_recording_is_optional_and_validates_teachers_before_saving(self):
        teacher = self.register_agent('Teacher')
        opponent = self.register_agent('Opponent')
        outsider = self.register_agent('Not in the game')
        agents = [teacher, opponent]
        ordinary = self.queue_run(agents)
        self.assertFalse(ordinary['config'].get('record_training', False))
        self.assertEqual(ordinary['config'].get('teachers', []), [])

        recorded = self.queue_run(agents, teachers=[teacher])
        self.assertIs(recorded['config']['record_training'], True)
        self.assertEqual(recorded['config']['teachers'], [teacher['id']])
        self.assertEqual(ordinary['profile'], recorded['profile'])

        body = {
            'mode': 'tournament',
            'agents': [agent['id'] for agent in agents],
            'seeds': ['validation-only'],
            'deals': 1,
            'record_training': True,
        }
        invalid_settings = [
            {},
            {'teachers': []},
            {'teachers': teacher['id']},
            {'teachers': None},
            {'teachers': [teacher['id'], teacher['id']]},
            {'teachers': [outsider['id']]},
            {'teachers': [42]},
            {'teachers': [teacher['id']], 'record_training': 'true'},
            {'teachers': [teacher['id']], 'record_training': 1},
        ]
        count_before = len(server.all_rows('runs'))
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    server.queue_run({**body, **settings})
                self.assertEqual(len(server.all_rows('runs')), count_before)

    def test_selected_teacher_records_exact_pre_action_observations(self):
        teacher = self.register_agent('Observed teacher', log_observations=True)
        opponents = [self.register_agent('Opponent one'), self.register_agent('Opponent two')]
        run = self.queue_run([teacher, *opponents], teachers=[teacher])
        result = server.game(run, [teacher, *opponents], 'recording-test:table', 1, 1,
                             seed_group='recording-test')
        history, records = self.read_game(run, result)
        self.assertEqual(result['verdicts'], ['OK', 'OK', 'OK'])
        self.assertGreater(len(records), 3)
        self.assertEqual(result['training']['records'], len(records))
        self.assertEqual(result['training']['teacher_counts'], {teacher['id']: len(records)})
        self.assertEqual(result['training']['excluded_records'], 0)

        received_observations = []
        for line in history['logs'][0].splitlines():
            if line.startswith('OBS:'):
                received_observations.append(json.loads(line.removeprefix('OBS:')))
        self.assertEqual([record['observation'] for record in records], received_observations)
        self.assertEqual(len({record['decision_id'] for record in records}), len(records))

        expected_observation_fields = {
            'type', 'hand', 'seat', 'street', 'board', 'hole', 'pot', 'to_call',
            'min_raise_to', 'max_raise_to', 'can_raise', 'stacks', 'street_bets',
            'folded', 'button', 'history', 'players', 'clock_ms',
        }
        for record in records:
            observation = record['observation']
            self.assertEqual(record['schema_version'], 1)
            self.assertEqual(record['run_id'], run['id'])
            self.assertEqual(record['game_id'], 1)
            self.assertEqual(record['seed_group'], 'recording-test')
            self.assertEqual(record['game_seed'], 'recording-test:table')
            self.assertEqual(record['offset'], 1)
            self.assertEqual(record['teacher'], {
                'id': teacher['id'], 'name': teacher['name'],
                'hash': teacher['hash'], 'player': 0,
            })
            self.assertEqual(record['verdict'], 'OK')
            self.assertGreaterEqual(record['elapsed_ms'], 0)
            self.assertEqual(set(observation), expected_observation_fields)
            self.assertEqual(record['requested_action'], {'action': 'call'})

            # Match only this actor's private hand, and public events preceding the action.
            hand = history['hands'][observation['hand']]
            seat = observation['seat']
            self.assertEqual(observation['players'][seat], 0)
            self.assertEqual(observation['hole'], hand['holes'][str(seat)])
            action_events = [event for event in hand['events'] if event['type'] == 'action']
            action_index = len(observation['history'])
            preceding_actions = action_events[:action_index]
            expected_history = [[event['street'], event['seat'], event['action'], event['amount']]
                                for event in preceding_actions]
            self.assertEqual(observation['history'], expected_history)
            executed = action_events[action_index]
            self.assertEqual(executed['seat'], seat)
            self.assertEqual(record['executed_action'], {
                'action': executed['action'], 'amount': executed['amount'],
            })
            self.assertEqual(observation['pot'], 600 - sum(observation['stacks']))
            visible_cards = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
            self.assertEqual(len(observation['board']), visible_cards[observation['street']])

    def test_requested_raise_is_distinct_from_engine_normalized_action(self):
        teacher = self.register_agent('Oversize raise', action='return state.raise_to(1000000)')
        opponent = self.register_agent('Caller')
        run = self.queue_run([teacher, opponent], teachers=[teacher], deals=2)
        result = server.game(run, [teacher, opponent], 'raise-normalization', 0, 1)
        _, records = self.read_game(run, result)
        self.assertTrue(records)
        self.assertEqual(result['verdicts'], ['OK', 'OK'])
        for record in records:
            self.assertEqual(record['requested_action'], {'action': 'raise', 'amount': 1000000})
            self.assertEqual(record['executed_action'], {'action': 'raise', 'amount': 200})

    def test_failed_teacher_entire_game_is_excluded_without_fallback_labels(self):
        failing = self.register_agent('Fails after one hand', action=(
            'if state.hand > 0:\n'
            '    raise RuntimeError("intentional recording test failure")\n'
            'return state.call()'
        ))
        healthy = self.register_agent('Healthy teacher')
        run = self.queue_run([failing, healthy], teachers=[failing, healthy], deals=3)
        result = server.game(run, [failing, healthy], 'late-failure', 0, 1)
        _, records = self.read_game(run, result)
        self.assertEqual(result['verdicts'], ['RTE', 'OK'])
        self.assertTrue(records)
        self.assertEqual({record['teacher']['id'] for record in records}, {healthy['id']})
        self.assertGreater(result['training']['excluded_records'], 0)
        self.assertEqual(result['training']['failed_teachers'], [failing['id']])
        self.assertEqual(result['training']['teacher_counts'].get(failing['id'], 0), 0)
        self.assertTrue(all(record['verdict'] == 'OK' for record in records))

    def test_recording_leaves_game_outcomes_unchanged_and_default_writes_no_dataset(self):
        teacher = self.register_agent('Caller')
        opponent = self.register_agent('Raiser', action='return state.raise_to(8)')
        agents = [teacher, opponent]
        ordinary = self.queue_run(agents, deals=4)
        recorded = self.queue_run(agents, teachers=[teacher, opponent], deals=4)
        ordinary_result = server.game(ordinary, agents, 'same-decks-and-policies', 0, 1)
        recorded_result = server.game(recorded, agents, 'same-decks-and-policies', 0, 1)
        ordinary_history, _ = self.read_game(ordinary, ordinary_result)
        recorded_history, records = self.read_game(recorded, recorded_result)
        self.assertTrue(records)
        for key in ['chips', 'verdicts', 'num_hands', 'mbb_per_hand', 'hands']:
            self.assertEqual(ordinary_history[key], recorded_history[key], key)
        self.assertNotIn('training', ordinary_result)
        ordinary_folder = server.DATA / 'runs' / ordinary['id']
        self.assertEqual(list(ordinary_folder.glob('*.jsonl')), [])

    def test_benchmark_records_only_selected_teacher_even_at_separate_tables(self):
        teacher = self.register_agent('Selected teacher')
        other = self.register_agent('Opponent only', action='return state.all_in()')
        run = self.queue_run([teacher, other], teachers=[teacher], deals=1, mode='benchmark')
        server.execute(run)
        teacher_records = 0
        for table in run['results']:
            table_teacher_records = 0
            for game in table['games']:
                _, records = self.read_game(run, game)
                table_teacher_records += len(records)
                for record in records:
                    self.assertEqual(record['teacher']['id'], teacher['id'])
            if table['agents'][0]['id'] == other['id']:
                self.assertEqual(table_teacher_records, 0)
            teacher_records += table_teacher_records
        self.assertGreater(teacher_records, 0)
        self.assertEqual(run['training']['records'], teacher_records)
        self.assertEqual(run['training']['teacher_counts'], {teacher['id']: teacher_records})

    def test_no_dataset_download_when_all_teachers_fail_before_acting(self):
        agents = [
            self.register_agent('Failed one', action='raise RuntimeError("expected failure")'),
            self.register_agent('Failed two', action='raise RuntimeError("expected failure")'),
        ]
        run = self.queue_run(agents, teachers=agents, deals=1)
        server.execute(run)
        self.assertEqual(run['status'], 'completed')
        self.assertEqual(run['training']['records'], 0)
        self.assertNotIn('download_url', run['training'])
        with self.assertRaisesRegex(ValueError, 'no valid teacher decisions'):
            server.training_archive(run['id'])

    def test_duplicate_rotations_export_together_and_old_runs_still_work(self):
        teacher = self.register_agent('Teacher')
        opponent = self.register_agent('Opponent')
        agents = [teacher, opponent]
        run = self.queue_run(agents, teachers=[teacher], deals=1)
        # The real tournament must retain the outer seed through rounds and playoffs.
        with self.assertRaises(ValueError):
            server.training_archive(run['id'])
        server.execute(run)
        records_by_game = []
        offsets = set()
        seed_groups = set()
        game_seeds = set()
        for table in run['results']:
            for game in table['games']:
                _, records = self.read_game(run, game)
                records_by_game.append(records)
                for record in records:
                    offsets.add(record['offset'])
                    seed_groups.add(record['seed_group'])
                    game_seeds.add(record['game_seed'])
        self.assertEqual(offsets, {0, 1})
        self.assertEqual(seed_groups, {'recording-test'})
        self.assertGreaterEqual(len(game_seeds), 4)
        self.assertEqual(run['training']['status'], 'ready')
        total_records = sum(len(records) for records in records_by_game)
        self.assertEqual(run['training']['records'], total_records)

        archive_path = server.training_archive(run['id'])
        with zipfile.ZipFile(archive_path) as archive:
            expected_files = {'manifest.json'}
            for game_number in range(1, len(records_by_game) + 1):
                expected_files.add(f'game-{game_number}.decisions.jsonl')
            self.assertEqual(set(archive.namelist()), expected_files)
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['schema_version'], 1)
            self.assertEqual(manifest['run_id'], run['id'])
            self.assertEqual(manifest['records'], total_records)
            self.assertEqual(manifest['teacher_counts'], {teacher['id']: total_records})
            self.assertEqual(manifest['teachers'], [{
                'id': teacher['id'], 'name': teacher['name'], 'hash': teacher['hash'],
            }])
            for game_number, expected in enumerate(records_by_game, start=1):
                exported = archive.read(f'game-{game_number}.decisions.jsonl').decode()
                self.assertEqual([json.loads(line) for line in exported.splitlines()], expected)
            for file_info in manifest['files']:
                content = archive.read(file_info['name'])
                self.assertEqual(file_info['bytes'], len(content))
                self.assertEqual(file_info['sha256'], hashlib.sha256(content).hexdigest())
        self.assertEqual(server.training_archive(run['id']), archive_path)

        old_run = self.queue_run(agents)
        old_run['config'].pop('record_training', None)
        old_run['config'].pop('teachers', None)
        server.save('runs', old_run)
        result = server.game(old_run, agents, 'old-run-compatible', 0, 1)
        self.assertEqual(result['verdicts'], ['OK', 'OK'])
        self.assertNotIn('training', result)
        self.assertTrue(server.state()['runs'])
        with self.assertRaises(ValueError):
            server.training_archive(old_run['id'])


if __name__ == '__main__':
    unittest.main()
