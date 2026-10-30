"""Small CPU checks for exact archives, recovery and truthful SDK benchmarks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from macpoker.match import MatchConfig, MatchRunner

from arena import ppo_milestones as milestones
from arena import ppo_train_worker as worker
from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION
from arena.bc_train_worker import build_model, model_layers
from arena.ppo_rollout import OpponentTransport
from arena.scoring import points
from arena.tests.test_ppo_training import fake_evaluate, fake_rollout, settings


class MilestoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.source = self.folder / 'source.pt'
        torch.manual_seed(42)
        torch.save({'model': build_model().state_dict(), 'epoch': 2}, self.source)
        self.opponent_folder = self.folder / 'caller'
        self.opponent_folder.mkdir()
        (self.opponent_folder / 'main.py').write_text(
            'from macpoker import Bot\nclass Caller(Bot):\n    def act(self, state):\n        return state.call()\n')
        self.opponent = {'id': 'caller', 'name': 'Caller', 'hash': 'test', 'path': str(self.opponent_folder)}
        worker.PAUSE_SIGNAL = False

    def tearDown(self):
        worker.PAUSE_SIGNAL = False
        self.temporary.cleanup()

    def request(self, name='job', **changes):
        folder = self.folder / name
        folder.mkdir()
        request = {'job_id': name, 'feature_version': FEATURE_VERSION,
                   'source_checkpoint': {'path': str(self.source), 'sha256': milestones.file_hash(self.source),
                                         'training_id': 'bc', 'epoch': 2, 'name': 'Source'},
                   'opponents': [self.opponent],
                   'config': settings(target_episodes=5, rollout_episodes=3, **changes)}
        worker.write_json(folder / 'request.json', request)
        return request, folder

    def activate(self, request, folder, interval=2):
        return milestones.ensure_settings(folder, request, {'interval': interval, 'eval_hands': 1, 'eval_seeds': 1})

    def fake_game(self, weights, game, work_dir, check_pause=None):
        if check_pause is not None:
            check_pause()
        # Deliberately consume global RNG so preservation is tested, not assumed.
        random.random()
        torch.rand(3)
        chips = [2, -2] + [0] * (game['table_size'] - 2)
        placement = points(chips)
        return {'completed': True, 'completed_hands': game['requested_hands'],
                'chips': chips, 'verdicts': ['OK'] * game['table_size'], 'points': placement,
                'rank': game['table_size'] + 1 - placement[0], 'normalized_placement': 1.0,
                'net_chips': 2, 'bb100': 100 / game['requested_hands'], 'error': None}

    def train_fake(self, request, folder, game=None):
        with patch('arena.ppo_rollout.collect_rollout', side_effect=fake_rollout):
            with patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate):
                with patch.object(milestones, 'play_benchmark_game', side_effect=game or self.fake_game):
                    worker.train(request, folder, {})

    def test_defaults_and_next_rollout_exact_boundaries(self):
        self.assertEqual(milestones.normalize_settings()['interval'], 50000)
        self.assertEqual(milestones.normalize_settings()['eval_hands'], 100)
        self.assertEqual(milestones.normalize_settings()['eval_seeds'], 3)
        self.assertEqual(milestones.next_rollout_count(49999, 130000, 3000, 50000), 1)
        self.assertEqual(milestones.next_rollout_count(50000, 130000, 3000, 50000), 3000)
        self.assertEqual(milestones.next_rollout_count(99800, 100001, 3000, 50000), 200)
        self.assertEqual(milestones.next_rollout_count(100000, 100001, 3000, 50000), 1)
        for raw in ({'eval_hands': True}, {'interval': 0}, {'table_sizes': [4]}, {'enabled': 1}, [], '', 0, True):
            with self.assertRaises(ValueError):
                milestones.normalize_settings(raw)

    def test_exact_snapshots_final_current_policy_and_unchanged_best_objective(self):
        request, folder = self.request()
        self.activate(request, folder)
        self.train_fake(request, folder)
        summary = milestones.summaries(folder)
        self.assertEqual([row['episodes'] for row in summary['snapshots']], [2, 4, 5])
        self.assertEqual([row['kind'] for row in summary['snapshots']], ['milestone', 'milestone', 'final'])
        self.assertEqual(summary['leaders'], {'bb100': 'bc-000000000', 'normalized_placement': 'bc-000000000'})
        final = torch.load(folder / 'milestones/final-000000005/checkpoint.pt', weights_only=False)
        latest = torch.load(folder / 'latest.pt', weights_only=False)
        best = torch.load(folder / 'best.pt', weights_only=False)
        self.assertEqual(best['progress']['best_iteration'], 0)
        self.assertEqual(best['progress']['best_score'], 3.0)
        self.assertEqual(final['progress']['episodes'], 5)
        self.assertTrue(final['actor_optimizer']['state'])
        self.assertEqual(final['progress'], latest['progress'])
        for name in final['actor']:
            self.assertTrue(torch.equal(final['actor'][name], latest['actor'][name]))
        baseline = milestones.snapshot(folder, 'bc-000000000')
        self.assertFalse(baseline['resumable'])
        self.assertEqual(baseline['checkpoint_format'], 'bc')
        self.assertEqual(milestones.file_hash(baseline['checkpoint_path']), milestones.file_hash(self.source))
        self.assertEqual(baseline['feature_count'], FEATURE_COUNT)
        for row in [summary['baseline']] + summary['snapshots']:
            self.assertEqual(row['completed_games'], 15)
            self.assertEqual(row['completed_hands'], 15)
            self.assertEqual(row['by_table_size']['6']['completed_games'], 6)

    def test_pause_in_milestone_resumes_games_without_retraining_saved_iteration(self):
        reference_request, reference_folder = self.request('reference')
        self.activate(reference_request, reference_folder)
        self.train_fake(reference_request, reference_folder)
        request, folder = self.request('resumed')
        self.activate(request, folder)
        calls = 0

        def pause_game(*arguments, **keywords):
            nonlocal calls
            calls += 1
            if calls == 18:
                random.random()
                torch.rand(7)
                raise worker.TrainingPaused()
            return self.fake_game(*arguments, **keywords)

        with self.assertRaises(worker.TrainingPaused):
            self.train_fake(request, folder, pause_game)
        saved = torch.load(folder / 'latest.pt', weights_only=False)
        self.assertEqual(saved['progress']['episodes'], 2)
        self.assertEqual(saved['progress']['iteration'], 1)
        partial_folder = folder / 'milestones/episode-000000002'
        self.assertEqual(len(list((partial_folder / 'games').glob('*.json'))), 2)
        partial_hashes = {path.name: milestones.file_hash(path) for path in (partial_folder / 'games').glob('*.json')}
        self.assertEqual(milestones.summaries(folder)['pending'], ['episode-000000002'])
        partial = milestones.summaries(folder)['snapshots'][0]
        self.assertEqual(partial['attempted_games'], 3)
        self.assertEqual(partial['completed_games'], 2)
        resumed_calls = []

        def resumed_game(weights, game, work_dir, check_pause=None):
            resumed_calls.append(game['game_id'])
            return self.fake_game(weights, game, work_dir, check_pause)

        self.train_fake(request, folder, resumed_game)
        self.assertEqual(len(resumed_calls), 13 + 15 + 15)
        for name, checksum in partial_hashes.items():
            self.assertEqual(milestones.file_hash(partial_folder / 'games' / name), checksum)
        actual = torch.load(folder / 'latest.pt', weights_only=False)
        expected = torch.load(reference_folder / 'latest.pt', weights_only=False)
        self.assertEqual(actual['progress'], expected['progress'])
        for component in ('actor', 'critic'):
            for name in actual[component]:
                self.assertTrue(torch.equal(actual[component][name], expected[component][name]))
        self.assertTrue(torch.equal(actual['torch_rng'], expected['torch_rng']))
        self.assertEqual(actual['python_rng'], expected['python_rng'])

    def test_legacy_request_hash_and_unrecoverable_past_milestones(self):
        request, folder = self.request()
        original_request = (folder / 'request.json').read_bytes()
        collected = 0

        def pause_second_rollout(*arguments, **keywords):
            nonlocal collected
            collected += 1
            if collected == 2:
                raise worker.TrainingPaused()
            return fake_rollout(*arguments, **keywords)

        with patch('arena.ppo_rollout.collect_rollout', side_effect=pause_second_rollout):
            with patch('arena.ppo_rollout.evaluate_policy', side_effect=fake_evaluate):
                with self.assertRaises(worker.TrainingPaused):
                    worker.train(request, folder, {})
        self.assertFalse((folder / 'milestone-settings.json').exists())
        checkpoint = torch.load(folder / 'latest.pt', weights_only=False)
        self.assertEqual(checkpoint['progress']['episodes'], 3)
        self.activate(request, folder)
        self.train_fake(request, folder)
        self.assertEqual(original_request, (folder / 'request.json').read_bytes())
        self.assertEqual(checkpoint['request_hash'], milestones.request_hash(request))
        summary = milestones.summaries(folder)
        self.assertEqual(summary['tracking_start_episodes'], 3)
        self.assertEqual(summary['unrecoverable_past_milestones'], [2])
        self.assertEqual([row['episodes'] for row in summary['snapshots']], [4, 5])

    def test_changed_sidecar_opponents_and_snapshot_bytes_are_rejected(self):
        request, folder = self.request()
        self.activate(request, folder)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            milestones.ensure_settings(folder, request, {'interval': 3})
        self.train_fake(request, folder)
        saved = milestones.snapshot(folder, 'episode-000000002')
        with Path(saved['checkpoint_path']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            milestones.snapshot(folder, 'episode-000000002')
        (self.opponent_folder / 'main.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'opponent checksum'):
            milestones.begin_tracking(folder, request, 5)
        with self.assertRaisesRegex(ValueError, 'Invalid milestone'):
            milestones.snapshot(folder, '../../latest')

    def test_ties_and_mixed_size_rates_use_equal_size_weighting(self):
        games = []
        for size, chip_total in ((4, 8), (5, 0), (6, -12)):
            chips = [chip_total, chip_total] + [-2 * chip_total / (size - 2)] * (size - 2)
            placement = points(chips)
            games.append({'table_size': size, 'completed': True, 'completed_hands': 100,
                          'requested_hands': 100, 'chips': chips, 'verdicts': ['OK'] * size,
                          'points': placement, 'rank': size + 1 - placement[0],
                          'normalized_placement': (placement[0] - 1) / (size - 1),
                          'net_chips': chips[0], 'error': None})
        self.assertEqual(games[0]['points'][:2], [3.5, 3.5])
        self.assertEqual(games[0]['rank'], 1.5)
        self.assertEqual(games[1]['points'], [3] * 5)
        manifest = {'id': 'test', 'request_hash': 'request', 'settings_sha256': 'settings',
                    'checkpoint_sha256': 'checkpoint', 'weights_sha256': 'weights'}
        report = milestones.report_from_games(manifest, milestones.normalize_settings(), games, [None] * 3)
        self.assertAlmostEqual(report['bb100'], (4 + 0 - 6) / 3)
        self.assertAlmostEqual(report['normalized_placement'], (2.5 / 3 + 2 / 4 + 0.5 / 5) / 3)
        games[0]['verdicts'][1] = 'RTE'
        totals = milestones.aggregate_games(games, 3)
        self.assertEqual(totals['opponent_failures'], 1)
        self.assertFalse(totals['eligible_for_best'])
        games[2].update(completed=False, completed_hands=37, error='interrupted', net_chips=None)
        totals = milestones.aggregate_games(games, 3)
        self.assertEqual(totals['attempted_hands'], 300)
        self.assertEqual(totals['completed_hands'], 237)
        self.assertEqual(totals['completed_games'], 2)
        self.assertEqual(totals['net_chips'], 8)

    def test_failure_reports_cannot_become_leaders_even_with_large_chip_scores(self):
        request, folder = self.request()
        self.activate(request, folder)

        def failed_game(weights, game, work_dir, check_pause=None):
            result = self.fake_game(weights, game, work_dir, check_pause)
            result['verdicts'][0] = 'RTE'
            result['net_chips'] = 100000
            return result

        self.train_fake(request, folder, failed_game)
        summary = milestones.summaries(folder)
        self.assertEqual(summary['leaders'], {'bb100': None, 'normalized_placement': None})
        self.assertEqual(summary['baseline']['status'], 'failed')
        self.assertEqual(summary['baseline']['learner_failures'], 15)
        self.assertEqual(summary['baseline']['completed_games'], 15)

    def test_immutable_archive_and_report_recovery_deduplicates_completed_work(self):
        request, folder = self.request()
        sidecar = self.activate(request, folder)
        self.train_fake(request, folder)
        before = {str(path.relative_to(folder)): milestones.file_hash(path)
                  for path in (folder / 'milestones').rglob('*') if path.is_file()}
        saved = torch.load(folder / 'latest.pt', weights_only=False)
        with patch.object(milestones, 'play_benchmark_game', side_effect=AssertionError('Completed games must not rerun')):
            milestones.checkpoint_hook(folder, request, saved['progress'], sidecar)
        after = {str(path.relative_to(folder)): milestones.file_hash(path)
                 for path in (folder / 'milestones').rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        destination = folder / 'milestones/final-000000005'
        (destination / 'manifest.json').unlink()
        with milestones.preserve_rng():
            recovered = milestones.archive(folder, request, folder / 'latest.pt', 'final', 5, saved['progress']['iteration'])
        self.assertEqual(recovered['id'], 'final-000000005')
        self.assertEqual(before['milestones/final-000000005/checkpoint.pt'], milestones.file_hash(recovered['checkpoint_path']))

    def test_sdk_full_game_continues_after_opponent_and_learner_failure(self):
        (self.opponent_folder / 'main.py').write_text(
            'from macpoker import Bot\nclass Broken(Bot):\n    def act(self, state):\n        raise RuntimeError("broken")\n')
        game = {'table_size': 4, 'requested_hands': 100, 'offset': 2,
                'seed': 'failure-test', 'opponents': [self.opponent] * 3}
        weights = {'layers': model_layers(build_model())}
        result = milestones.play_benchmark_game(weights, game, self.folder / 'work')
        self.assertTrue(result['completed'])
        self.assertEqual(result['completed_hands'], 100)
        self.assertEqual(result['verdicts'][0], 'OK')
        self.assertGreater(sum(value != 'OK' for value in result['verdicts'][1:]), 0)
        self.assertEqual(result['points'], points(result['chips']))
        self.assertEqual(result['net_chips'], result['chips'][0])
        bad_weights = deepcopy(weights)
        bad_weights['layers'][-1]['bias'][0] = float('nan')
        result = milestones.play_benchmark_game(bad_weights, {**game, 'requested_hands': 2}, self.folder / 'work')
        self.assertEqual(result['verdicts'][0], 'RTE')
        self.assertEqual(result['completed_hands'], 2)
        self.assertEqual(list((self.folder / 'work').iterdir()), [])

    def test_recover_report_checksum_crash_window_from_durable_games(self):
        request, folder = self.request()
        sidecar = self.activate(request, folder)
        self.train_fake(request, folder)
        destination = folder / 'milestones/final-000000005'
        report_bytes = (destination / 'report.json').read_bytes()
        (destination / 'report-checksum.json').unlink()
        self.assertFalse(milestones.summaries(folder)['snapshots'][-1]['eligible_for_best'])
        manifest = milestones.snapshot(folder, 'final-000000005')
        with patch.object(milestones, 'play_benchmark_game', side_effect=AssertionError('Durable games must not rerun')):
            milestones.evaluate_snapshot(folder, manifest, sidecar)
        self.assertEqual((destination / 'report.json').read_bytes(), report_bytes)
        self.assertTrue((destination / 'report-checksum.json').is_file())
        self.assertTrue(milestones.summaries(folder)['snapshots'][-1]['eligible_for_best'])
        # With no completion marker, fabricated scores must still disagree
        # with the reconstructed immutable report and stop recovery.
        (destination / 'report-checksum.json').unlink()
        tampered = json.loads(report_bytes)
        tampered['bb100'] = 999999
        (destination / 'report.json').write_text(json.dumps(tampered))
        with self.assertRaisesRegex(ValueError, 'Immutable milestone artifact'):
            milestones.evaluate_snapshot(folder, manifest, sidecar)

    def test_serialized_inprocess_inference_matches_exported_sdk_subprocess(self):
        import sys
        game = {'table_size': 4, 'requested_hands': 3, 'offset': 1,
                'seed': 'serialization-test', 'opponents': [self.opponent] * 3}
        weights = {'layers': model_layers(build_model())}
        actual = milestones.play_benchmark_game(weights, game, self.folder / 'work')
        learner_folder = self.folder / 'exported'
        learner_folder.mkdir()
        from arena.cloning import playable_files
        for name, content in playable_files('Test export', json.dumps(weights).encode()).items():
            (learner_folder / name).write_bytes(content)
        runner_path = Path(worker.__file__).with_name('bot_runner.py')
        transports = []
        try:
            for player in range(4):
                path = learner_folder if player == 0 else self.opponent_folder
                seed = f"{game['seed']}:learner:{game['offset']}" if player == 0 else f"{game['seed']}:opponent:{player}"
                transports.append(OpponentTransport([sys.executable, '-B', '-u', str(runner_path), str(path / 'main.py'), seed]))
            result = MatchRunner(MatchConfig(seats=4, deals=3, offset=1, seed=game['seed']), transports).run()
        finally:
            for transport in transports:
                transport.close()
        self.assertEqual(result.verdicts, ['OK'] * 4)
        self.assertEqual(actual['verdicts'], result.verdicts)
        self.assertEqual(actual['chips'], result.chips)
        self.assertEqual(actual['points'], points(result.chips))


if __name__ == '__main__':
    unittest.main()
