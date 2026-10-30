"""Shared milestone exports identify the optimizer that can resume them."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from arena import ppo_milestones
from arena.bc_features import FEATURE_VERSION
from arena.bc_train_worker import build_model


class MilestoneMetadataTests(unittest.TestCase):
    def test_grpo_archive_is_not_labelled_as_a_ppo_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            actor = build_model()
            source = folder / 'source_bc.pt'
            torch.save({'model': actor.state_dict()}, source)
            request = {'source_checkpoint': {'path': str(source), 'sha256': ppo_milestones.file_hash(source)},
                       'config': {'reward_mode': 'chips'}}
            (folder / 'request.json').write_text(json.dumps(request))
            (folder / 'milestone-settings.json').write_text(json.dumps({'enabled': True}))
            path = folder / 'latest.pt'
            checkpoint = {
                'actor': actor.state_dict(), 'algorithm': 'grpo',
                'feature_version': FEATURE_VERSION, 'progress': {'episodes': 400},
                'request_hash': ppo_milestones.request_hash(request),
            }
            torch.save(checkpoint, path)
            manifest = ppo_milestones.archive(folder, request, path, 'milestone', 400, 1)
            self.assertEqual(manifest['checkpoint_format'], 'grpo')
            self.assertTrue(manifest['resumable'])


if __name__ == '__main__':
    unittest.main()
