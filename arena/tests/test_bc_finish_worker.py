"""Tiny CPU tests for finishing BC with its best validated checkpoint."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from arena import bc_train_worker as training
from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION
from arena.bc_features import action_options, encode_observation, forward_logits


class FinishWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.config = {'teachers': ['teacher'], 'table_sizes': [4], 'epochs': 5,
                       'batch_size': 4, 'learning_rate': 0.001, 'balanced': True,
                       'teacher_weights': {}, 'device': 'cpu'}
        self.request = {'config': self.config}
        observation = {'seat': 0, 'street': 'preflop', 'hole': ['As', 'Kd'], 'board': [],
                       'pot': 3, 'to_call': 2, 'min_raise_to': 4, 'max_raise_to': 200,
                       'can_raise': True, 'stacks': [200, 199, 198, 200],
                       'street_bets': [0, 1, 2, 0], 'folded': [False] * 4,
                       'button': 0, 'history': []}
        self.features = encode_observation(observation)
        _, mask = action_options(observation)
        dtype = np.dtype([('features', 'f4', (FEATURE_COUNT,)), ('mask', '?', (7,)),
                          ('target', 'u1'), ('teacher', 'u2'), ('size', 'u1'), ('split', 'u1')])
        self.array = np.zeros(18, dtype=dtype)
        self.array['features'] = self.features
        self.array['mask'] = mask
        self.array['target'] = np.arange(18) % 2
        self.array['size'] = 4
        self.array['split'][12:15] = 1
        self.array['split'][15:] = 2
        self.metadata = {'records': 18, 'duplicate_records': 0,
                         'split_counts': {'train': 12, 'validation': 3, 'test': 3},
                         'raise_approximation': {'raises': 0, 'mean_absolute_chips': 0},
                         'teacher_ids': ['teacher']}

    def tearDown(self):
        self.temporary.cleanup()

    def history(self, epochs):
        return [{'epoch': epoch, 'train_loss': 1.0, 'train_accuracy': 0.5,
                 'validation_loss': 1.0, 'validation_accuracy': 0.5}
                for epoch in range(1, epochs + 1)]

    def checkpoints(self, latest_epoch=2, best_epoch=1):
        model = training.build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.01)
        training.save_checkpoint(self.folder / 'best.pt', model, optimizer, best_epoch,
                                 self.history(best_epoch), 1.0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.02)
        training.save_checkpoint(self.folder / 'latest.pt', model, optimizer, latest_epoch,
                                 self.history(latest_epoch), 1.0)

    def train(self, report=None):
        if report is None:
            report = {'status': 'running', 'history': []}
        with patch.object(training, 'prepare_data', return_value=(self.array, self.metadata)):
            training.train(self.request, self.folder, report)
        return report

    def finish(self):
        training.write_json(self.folder / 'control.json', {'action': 'finish'})

    def assert_export_is_best(self):
        checkpoint = torch.load(self.folder / 'best.pt', weights_only=False)
        best = training.build_model()
        best.load_state_dict(checkpoint['model'])
        weights = json.loads((self.folder / 'weights.json').read_text())
        self.assertEqual(weights['layers'], training.model_layers(best))
        expected = best(torch.tensor(self.features)).detach().tolist()
        exported = forward_logits(self.features, weights['layers'])
        for left, right in zip(expected, exported):
            self.assertAlmostEqual(left, right, places=5)
        self.assertEqual(weights['feature_version'], FEATURE_VERSION)
        self.assertEqual(weights['best_epoch'], checkpoint['epoch'])

    def test_finish_saved_checkpoint_takes_no_optimizer_steps_and_exports_best(self):
        self.checkpoints()
        self.finish()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Unexpected optimization')):
            report = self.train()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['epoch'], 2)
        self.assertEqual(report['epochs'], 5)
        self.assertEqual(report['best_epoch'], 1)
        self.assertTrue(report['stopped_early'])
        self.assertFalse(report['finish_requested'])
        self.assertEqual(report['history'], self.history(2))
        self.assertEqual(set(report['metrics']), {'validation', 'test'})
        self.assert_export_is_best()

    def test_mid_epoch_finish_discards_partial_epoch_and_preserves_history(self):
        self.checkpoints()
        original_step = torch.optim.Adam.step
        steps = []
        def step_and_finish(optimizer, *arguments, **keywords):
            result = original_step(optimizer, *arguments, **keywords)
            steps.append(1)
            self.finish()
            return result
        with patch.object(torch.optim.Adam, 'step', new=step_and_finish):
            report = self.train()
        self.assertEqual(len(steps), 1)
        self.assertEqual(report['epoch'], 2)
        self.assertEqual(report['history'], self.history(2))
        saved = torch.load(self.folder / 'latest.pt', weights_only=False)
        self.assertEqual(saved['epoch'], 2)
        self.assert_export_is_best()

    def test_candidate_validation_can_finish_without_committing_partial_metrics(self):
        self.checkpoints()
        original_evaluate = training.evaluate
        def finish_in_candidate(*arguments, **keywords):
            if keywords.get('allow_finish'):
                self.finish()
            return original_evaluate(*arguments, **keywords)
        with patch.object(training, 'evaluate', side_effect=finish_in_candidate):
            report = self.train()
        self.assertEqual(report['epoch'], 2)
        self.assertEqual(report['history'], self.history(2))
        self.assert_export_is_best()

    def test_finish_before_first_checkpoint_completes_one_validated_epoch(self):
        self.finish()
        report = self.train()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['epoch'], 1)
        self.assertEqual(report['best_epoch'], 1)
        self.assertEqual(len(report['history']), 1)
        self.assertTrue(report['stopped_early'])
        self.assert_export_is_best()

    def test_pause_during_finalization_then_resume_never_optimizes_again(self):
        self.checkpoints()
        self.finish()
        original_evaluate = training.evaluate
        report = {'status': 'running', 'history': []}
        def pause_final_evaluation(*arguments, **keywords):
            training.write_json(self.folder / 'control.json', {'action': 'pause'})
            return original_evaluate(*arguments, **keywords)
        with patch.object(training, 'evaluate', side_effect=pause_final_evaluation):
            with self.assertRaises(training.TrainingPaused):
                self.train(report)
        self.assertEqual(report['phase'], 'Finalizing best checkpoint')
        self.assertTrue(report['finish_requested'])
        self.assertEqual(report['epoch'], 2)
        self.assertFalse((self.folder / 'weights.json').exists())
        # The manager preserves finish intent and restores this control on Resume.
        self.finish()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Unexpected optimization')):
            completed = self.train()
        self.assertEqual(completed['status'], 'completed')
        self.assertTrue(completed['stopped_early'])
        self.assert_export_is_best()

    def test_newer_validated_best_survives_gap_before_latest_checkpoint_write(self):
        self.checkpoints(latest_epoch=0, best_epoch=1)
        self.finish()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Unexpected optimization')):
            report = self.train()
        self.assertEqual(report['epoch'], 1)
        self.assertEqual(report['history'], self.history(1))
        self.assert_export_is_best()

    def test_epoch_zero_best_is_never_exported(self):
        self.checkpoints(latest_epoch=0, best_epoch=0)
        self.finish()
        with self.assertRaisesRegex(ValueError, 'fully validated epoch'):
            self.train()
        self.assertFalse((self.folder / 'weights.json').exists())

    def test_natural_completion_keeps_requested_epochs_and_shared_actor_format(self):
        self.config['epochs'] = 2
        report = self.train()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['epoch'], 2)
        self.assertEqual(report['epochs'], 2)
        self.assertFalse(report['stopped_early'])
        self.assertFalse(report['finish_requested'])
        self.assertEqual(len(report['history']), 2)
        self.assert_export_is_best()
        self.assertEqual(str(training.get_device('cpu')), 'cpu')
        self.assertEqual(training.build_model()[0].in_features, FEATURE_COUNT)
        self.assertEqual(training.build_model()[-1].out_features, 7)


if __name__ == '__main__':
    unittest.main()
