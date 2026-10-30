"""PHH import checkpoints, provenance, controls and BC compatibility."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from arena import cloning, datasets, phh_jobs, server
from arena.tests.test_cloning import observation


class PHHJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_data = server.DATA
        self.previous_development = server.DEVELOPMENT
        server.DATA = self.root / 'dashboard-data'
        server.DEVELOPMENT = True
        server.STOP.clear()
        server.PAUSE.clear()
        server.init()
        self.source = self.root / 'source'
        for session in ('session-a', 'session-b', 'session-c'):
            folder = self.source / session
            folder.mkdir(parents=True)
            for number in range(4):
                (folder / f'{number}.phh').write_text(f'variant="NT"\nidentity="{session}-{number}"\n')
        self.convert_patch = mock.patch.object(phh_jobs, 'convert', side_effect=self.fake_convert)
        self.convert_patch.start()

    def tearDown(self):
        self.convert_patch.stop()
        server.DATA = self.previous_data
        server.DEVELOPMENT = self.previous_development
        server.STOP.clear()
        server.PAUSE.clear()
        self.temporary.cleanup()

    def body(self, **changes):
        body = {'name': 'PHH fixture', 'source_path': str(self.source), 'collection': 'folder',
                'teacher_name': 'Pluribus', 'max_hands': 12, 'profitable_only': False}
        body.update(changes)
        return body

    def fake_convert(self, hand, error, source_id, group_id, config):
        if error:
            return {'records': [], 'teachers': [], 'skip_reason': error, 'counters': {}}
        if config['profitable_only'] and not hand['identity'].endswith('0'):
            return {'records': [], 'teachers': [], 'skip_reason': 'not_profitable', 'counters': {}}
        teacher = {'id': 'phh-teacher', 'name': 'Pluribus', 'hash': 'fixed-teacher-version'}
        record = {'teacher': teacher, 'observation': observation(), 'verdict': 'OK',
                  'table_size': 4, 'decision_id': source_id, 'seed_group': group_id,
                  'requested_action': {'action': 'call'}, 'executed_action': {'action': 'call'}}
        return {'records': [record], 'teachers': [teacher], 'skip_reason': None,
                'counters': {'records': 1}}

    def records(self, job):
        records = []
        for shard in datasets.shards(server, job['id']):
            with gzip.open(server.DATA / 'datasets' / job['id'] / shard['name'], 'rt') as stream:
                records.extend(json.loads(line) for line in stream)
        return records

    def run_job(self, **changes):
        job = phh_jobs.queue(server, self.body(**changes))
        datasets.execute(server, job)
        return server.get('datasets', job['id'])

    def test_preview_is_bounded_read_only_and_interleaves_sessions(self):
        result = phh_jobs.preview(self.body(max_hands=3))
        self.assertEqual(result['sample_hands'], 3)
        self.assertEqual(result['records'], 3)
        self.assertEqual(server.all_rows('datasets'), [])
        self.assertEqual(server.all_rows('training_jobs'), [])
        config = phh_jobs.settings(self.body(max_hands=3))
        parents = {str(Path(entry['path']).parent) for entry in phh_jobs.source_files(config)}
        self.assertEqual(len(parents), 3)

    def test_import_is_native_bc_dataset_without_fake_registered_agents(self):
        job = self.run_job()
        self.assertEqual((job['status'], job['episodes'], job['records']), ('ready', 12, 12))
        self.assertEqual(len(job['seed_groups']), 3)
        self.assertEqual(server.all_rows('agents'), [])
        manifest_path = server.DATA / 'datasets' / job['id'] / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest['source'], 'PHH')
        self.assertEqual(len(manifest['source_fingerprints']), 12)
        self.assertEqual(manifest['phh_stats']['imported_hands'], 12)
        student = cloning.queue(server, {'name': 'Student', 'dataset_ids': [job['id']], 'device': 'auto'})
        self.assertEqual(student['config']['device'], 'cpu')
        self.assertEqual(student['config']['cpu_threads'], 1)
        with self.assertRaisesRegex(ValueError, 'CPU only'):
            cloning.queue(server, {'name': 'GPU denied', 'dataset_ids': [job['id']], 'device': 'mps'})
        self.assertTrue(datasets.archive(server, job['id']).is_file())

    def test_pause_resume_has_no_missing_or_duplicate_decisions(self):
        job = phh_jobs.queue(server, self.body())
        original = phh_jobs.convert
        calls = 0

        def pause_after_three(*args):
            nonlocal calls
            calls += 1
            result = original(*args)
            if calls == 3:
                server.PAUSE.add(job['id'])
            return result

        with mock.patch.object(phh_jobs, 'convert', side_effect=pause_after_three):
            datasets.execute(server, job)
        paused = server.get('datasets', job['id'])
        self.assertEqual((paused['status'], paused['records']), ('paused', 3))
        resumed = datasets.control(server, job['id'], 'resume')
        datasets.execute(server, resumed)
        final = server.get('datasets', job['id'])
        records = self.records(final)
        self.assertEqual((final['status'], final['processed_hands']), ('ready', 12))
        self.assertEqual(len({record['decision_id'] for record in records}), 12)

    def test_resumed_training_is_forced_to_cpu_at_launch(self):
        dataset = self.run_job()
        job = cloning.queue(server, {'name': 'Old GPU request', 'dataset_ids': [dataset['id']]})
        folder = server.DATA / 'training_runs' / job['id']
        request = json.loads((folder / 'request.json').read_text())
        request['config'].update(device='mps', cpu_threads=4)
        cloning.write_json(folder / 'request.json', request)
        job['config'].update(device='mps', cpu_threads=4)
        server.save('training_jobs', job)

        def check_launch(*args, **kwargs):
            request = json.loads((folder / 'request.json').read_text())
            self.assertEqual((request['config']['device'], request['config']['cpu_threads']), ('cpu', 1))
            cloning.write_json(folder / 'status.json', {'status': 'paused'})
            process = mock.Mock()
            process.poll.return_value = 0
            return process

        with mock.patch.object(cloning.subprocess, 'Popen', side_effect=check_launch), \
                mock.patch.object(server, 'kill_tree'):
            result = cloning.execute(server, job)
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(result['config']['device'], 'cpu')

    def test_finish_keeps_completed_hands(self):
        job = phh_jobs.queue(server, self.body())
        original = phh_jobs.convert
        calls = 0

        def finish_after_three(*args):
            nonlocal calls
            calls += 1
            result = original(*args)
            if calls == 3:
                datasets.control(server, job['id'], 'finish')
            return result

        with mock.patch.object(phh_jobs, 'convert', side_effect=finish_after_three):
            datasets.execute(server, job)
        result = server.get('datasets', job['id'])
        self.assertEqual((result['status'], result['records'], result['processed_hands']), ('ready', 3, 3))

    def test_filtering_and_reimport_keep_identical_decision_content(self):
        full = self.run_job()
        winners = self.run_job(profitable_only=True)
        full_records = {record['decision_id']: record for record in self.records(full)}
        self.assertEqual(winners['records'], 3)
        self.assertEqual(winners['processed_hands'], 12)
        self.assertEqual(winners['phh_stats']['skipped'], {'not_profitable': 9})
        for record in self.records(winners):
            self.assertEqual(record, full_records[record['decision_id']])

    def test_source_changes_refuse_resume(self):
        job = phh_jobs.queue(server, self.body())
        path = self.source / 'session-a' / '0.phh'
        path.write_text(path.read_text() + 'changed=true\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            datasets.execute(server, job)
        self.assertEqual(server.get('datasets', job['id'])['records'], 0)

    def test_finishing_saved_records_does_not_need_remaining_source_files(self):
        job = phh_jobs.queue(server, self.body())
        original = phh_jobs.convert

        def pause_after_one(*args):
            result = original(*args)
            server.PAUSE.add(job['id'])
            return result

        with mock.patch.object(phh_jobs, 'convert', side_effect=pause_after_one):
            datasets.execute(server, job)
        for path in self.source.rglob('*.phh'):
            path.unlink()
        job = datasets.control(server, job['id'], 'finish')
        datasets.execute(server, job)
        final = server.get('datasets', job['id'])
        self.assertEqual((final['status'], final['records']), ('ready', 1))

    def test_pluribus_session_import_preserves_collection_identity(self):
        repository = self.root / 'phh-dataset'
        session = repository / 'data' / 'pluribus' / '61'
        session.mkdir(parents=True)
        (session / '1.phh').write_text('variant="NT"\nidentity="same-hand"\n')
        full = phh_jobs.settings(self.body(source_path=str(repository), collection='pluribus'))
        partial = phh_jobs.settings(self.body(source_path=str(session), collection='folder'))
        self.assertEqual(full['source_namespace'], partial['source_namespace'])
        full_hand = next(phh_jobs.iter_hands(full, phh_jobs.source_files(full)))
        partial_hand = next(phh_jobs.iter_hands(partial, phh_jobs.source_files(partial)))
        self.assertEqual(full_hand[2:4], partial_hand[2:4])

    def test_invalid_settings_create_no_jobs(self):
        for changes in ({'max_hands': True}, {'max_hands': 0}, {'profitable_only': 'yes'},
                        {'collection': 'unknown'}, {'source_path': '/missing-phh-source'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                phh_jobs.queue(server, self.body(**changes))
        self.assertEqual(server.all_rows('datasets'), [])

    def test_phhs_and_malformed_files_have_explicit_counts(self):
        for path in self.source.rglob('*.phh'):
            path.unlink()
        (self.source / 'archive.phhs').write_text('[1]\nidentity="hand-0"\n[2]\nidentity="hand-1"\n')
        (self.source / 'broken.phh').write_text('not = valid = toml')
        result = self.run_job()
        self.assertEqual((result['processed_hands'], result['records']), (3, 2))
        self.assertEqual(result['phh_stats']['skipped'], {'malformed_phh': 1})


if __name__ == '__main__':
    unittest.main()
