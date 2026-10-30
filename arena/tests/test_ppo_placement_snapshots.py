"""Context-aware placement actors retain their objective through every export."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from arena import cloning, ppo_milestones, ppo_snapshots, ppo_snapshot_worker
from arena.bc_features import FEATURE_VERSION
from arena.bc_train_worker import build_model, model_layers
from arena.ppo_models import build_actor
from arena.ppo_context import CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT


class PlacementSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name).resolve()
        self.key = 'e' * 32
        self.folder = self.data / 'ppo_runs' / self.key
        self.folder.mkdir(parents=True)
        self.config = {'reward_mode': 'game_placement', 'placement_weight': 0.9,
                       'chip_scale_bb100': 100, 'seed': 42, 'target_episodes': 1000}
        self.job = {'id': self.key, 'name': 'Placement learner', 'status': 'paused', 'config': self.config}
        self.server = SimpleNamespace(DATA=self.data, LIVE_PPO_DATA=None, get=lambda table, key: self.job)
        source = self.folder / 'source_bc.pt'
        torch.save({'model': build_model().state_dict(), 'epoch': 1}, source)
        self.request = {'job_id': self.key, 'feature_version': FEATURE_VERSION, 'config': self.config,
                        'source_checkpoint': {'path': str(source), 'sha256': ppo_milestones.file_hash(source)},
                        'opponents': []}
        cloning.write_json(self.folder / 'request.json', self.request)
        actor = build_actor(context=True)
        self.checkpoint = {
            'actor': actor.state_dict(), 'critic': {}, 'actor_optimizer': {}, 'critic_optimizer': {},
            'torch_rng': torch.get_rng_state(), 'python_rng': (),
            'request_hash': ppo_milestones.request_hash(self.request),
            'feature_version': FEATURE_VERSION, 'feature_count': PLACEMENT_FEATURE_COUNT,
            'context_version': CONTEXT_VERSION, 'reward_mode': 'game_placement', 'reward_config': self.config,
            'selection_metric': 'placement_chip_score',
            'progress': {'episodes': 500, 'iteration': 1, 'best_iteration': 1, 'best_score': 0.73,
                         'selection_metric': 'placement_chip_score', 'best_selection_metric': 'placement_chip_score',
                         'validation': {'selection_score': 0.73, 'bb100': 12.5}},
        }
        torch.save(self.checkpoint, self.folder / 'best.pt')

    def tearDown(self):
        self.temporary.cleanup()

    def fast_worker(self, folder):
        validation = [{'table_size': size, 'hands': 2, 'verdicts': ['OK'] * size} for size in range(2, 7)]
        with patch.object(ppo_snapshot_worker, 'validate_games', return_value=validation):
            return ppo_snapshot_worker.export(folder)

    def test_saved_checkpoint_selects_objective_and_context_for_real_playable_zip(self):
        before = (self.folder / 'best.pt').read_bytes()
        result = ppo_snapshots.prepare(self.server, self.key)
        self.assertEqual(result['score'], 0.73)
        self.assertEqual(result['selection_metric'], 'placement_chip_score')
        self.assertEqual(result['feature_count'], PLACEMENT_FEATURE_COUNT)
        self.assertEqual(result['context_version'], CONTEXT_VERSION)
        self.assertEqual(result['reward_mode'], 'game_placement')
        self.assertEqual(result['reward_config']['placement_weight'], 0.9)
        self.assertEqual((self.folder / 'best.pt').read_bytes(), before)
        archive_path = ppo_snapshots.artifact(self.server, result['snapshot_id'])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(set(archive.namelist()), {'main.py', 'bc_features.py', 'ppo_context.py', 'weights.json'})
            self.assertNotIn('import torch', archive.read('main.py').decode())
            self.assertNotIn('import torch', archive.read('ppo_context.py').decode())
            weights = json.loads(archive.read('weights.json'))
            self.assertEqual(weights['feature_count'], PLACEMENT_FEATURE_COUNT)
            self.assertEqual(weights['context_version'], CONTEXT_VERSION)
        self.assertTrue(all(set(check['verdicts']) == {'OK'} for check in result['validation']))

    def test_baseline_archive_stays_legacy_while_placement_archive_has_context(self):
        ppo_milestones.ensure_settings(self.folder, self.request)
        baseline = ppo_milestones.archive(self.folder, self.request,
            Path(self.request['source_checkpoint']['path']), 'baseline', 0, 0)
        milestone = ppo_milestones.archive(self.folder, self.request, self.folder / 'best.pt', 'milestone', 500, 1)
        self.assertEqual(baseline['feature_count'], 189)
        self.assertNotIn('context_version', baseline)
        self.assertEqual(baseline['benchmark_reward_config']['reward_mode'], 'game_placement')
        self.assertEqual(milestone['feature_count'], PLACEMENT_FEATURE_COUNT)
        self.assertEqual(milestone['context_version'], CONTEXT_VERSION)
        with patch.object(ppo_snapshots, 'run_worker', side_effect=self.fast_worker):
            result = ppo_snapshots.prepare(self.server, self.key, kind='milestone', milestone_id=baseline['id'])
        with zipfile.ZipFile(ppo_snapshots.artifact(self.server, result['snapshot_id'])) as archive:
            self.assertNotIn('ppo_context.py', archive.namelist())
        self.assertFalse(result['resumable'])

    def test_lightweight_availability_does_not_claim_a_checkpoint_selection_metric(self):
        result = ppo_snapshots.describe(self.server, self.key)
        self.assertIsNone(result['best']['selection_metric'])
        self.assertEqual(result['best']['metadata_source'], 'verified_checkpoint_on_export')

    def test_context_milestone_inference_matches_exported_sdk_bot(self):
        import sys
        from macpoker.match import MatchConfig, MatchRunner
        from arena.ppo_rollout import OpponentTransport
        weights = {'feature_version': FEATURE_VERSION, 'feature_count': PLACEMENT_FEATURE_COUNT,
                   'context_version': CONTEXT_VERSION, 'layers': model_layers(build_actor(context=True))}
        opponent_folder = self.data / 'opponent'
        opponent_folder.mkdir()
        (opponent_folder / 'main.py').write_text(
            'from macpoker import Bot\nclass Caller(Bot):\n    def act(self, state):\n        return state.call()\n')
        opponent = {'id': 'caller', 'path': str(opponent_folder)}
        game = {'table_size': 4, 'requested_hands': 3, 'seed': 'context-parity',
                'offset': 2, 'opponents': [opponent] * 3}
        actual = ppo_milestones.play_benchmark_game(weights, game, self.data / 'work')
        bot_folder = self.data / 'exported'
        bot_folder.mkdir()
        for name, content in cloning.playable_files('Context test', json.dumps(weights).encode()).items():
            (bot_folder / name).write_bytes(content)
        transports = []
        runner_path = Path(cloning.__file__).with_name('bot_runner.py')
        try:
            for player in range(4):
                folder = bot_folder if player == 0 else opponent_folder
                seed = f"{game['seed']}:learner:{game['offset']}" if player == 0 else f"{game['seed']}:opponent:{player}"
                transports.append(OpponentTransport([sys.executable, '-B', '-u', str(runner_path), str(folder / 'main.py'), seed]))
            expected = MatchRunner(MatchConfig(seats=4, deals=3, offset=2, seed=game['seed']), transports).run()
        finally:
            for transport in transports:
                transport.close()
        self.assertEqual(actual['verdicts'], ['OK'] * 4)
        self.assertEqual(actual['chips'], expected.chips)

    def test_new_milestone_reports_and_leaders_keep_the_exact_round_objective(self):
        from arena.ppo_objectives import game_reward
        from arena.tests.test_ppo_placement_objectives import game
        sidecar = ppo_milestones.ensure_settings(self.folder, self.request, {'eval_seeds': 1})
        baseline = ppo_milestones.archive(self.folder, self.request,
            Path(self.request['source_checkpoint']['path']), 'baseline', 0, 0)
        milestone = ppo_milestones.archive(self.folder, self.request, self.folder / 'best.pt', 'milestone', 500, 1)
        for manifest in (baseline, milestone):
            games = []
            for size in (4, 5, 6):
                chips = [0] * size if manifest['kind'] == 'baseline' else [size - 1] + [-1] * (size - 1)
                for offset in range(size):
                    saved_game = game(chips, offset)
                    metrics = game_reward(chips)
                    saved_game.update(points=metrics['points'], normalized_placement=metrics['normalized_placement'],
                                      rank=size + 1 - metrics['points'][0], net_chips=chips[0], bb100=metrics['bb100'])
                    games.append(saved_game)
            report = ppo_milestones.report_from_games(manifest, sidecar['settings'], games, games)
            self.assertEqual(report['schema_version'], 2)
            self.assertEqual(report['selection_metric'], 'placement_chip_score')
            self.assertEqual(report['round_metrics']['completed_rounds'], 3)
            destination = self.folder / 'milestones' / manifest['id']
            ppo_milestones.atomic_json(destination / 'report.json', report, immutable=True)
            ppo_milestones.atomic_json(destination / 'report-checksum.json',
                {'sha256': ppo_milestones.file_hash(destination / 'report.json')}, immutable=True)
        summary = ppo_milestones.summaries(self.folder)
        self.assertEqual(summary['round_leaders']['placement_chip_score'], milestone['id'])
        self.assertEqual(summary['leaders']['placement_chip_score'], milestone['id'])


if __name__ == '__main__':
    unittest.main()
