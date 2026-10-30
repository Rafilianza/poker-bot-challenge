"""Public score memory, BC warm starts and SDK exports for placement policies."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import torch

from arena import cloning, ppo_snapshot_worker, ppo_snapshots
from arena.bc_features import FEATURE_COUNT, FEATURE_VERSION, encode_observation
from arena.bc_train_worker import build_model, model_layers
from arena.ppo_context import CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT, GameContext
from arena.ppo_models import build_actor, build_critic


def observation(players, seat, hand=0):
    count = len(players)
    return {
        'seat': seat, 'players': players, 'hand': hand, 'street': 'preflop',
        'hole': ['As', 'Kd'], 'board': [], 'pot': 3, 'to_call': 2,
        'min_raise_to': 4, 'max_raise_to': 200, 'can_raise': True,
        'stacks': [200] * count, 'street_bets': [0] * count,
        'folded': [False] * count, 'button': 0, 'history': [],
    }


def placement_checkpoint():
    actor = build_actor(build_model().state_dict(), context=True)
    # Nonzero context weights ensure export validation exercises the extra inputs.
    with torch.no_grad():
        actor[0].weight[:, FEATURE_COUNT:] = 0.25
    return {
        'actor': actor.state_dict(), 'critic': build_critic(context=True).state_dict(),
        'actor_optimizer': {}, 'critic_optimizer': {},
        'feature_version': FEATURE_VERSION, 'feature_count': PLACEMENT_FEATURE_COUNT,
        'context_version': CONTEXT_VERSION, 'reward_mode': 'game_placement',
        'selection_metric': 'placement_chip_score',
        'reward_config': {'reward_mode': 'game_placement', 'placement_weight': 0.9,
                          'chip_scale_bb100': 100, 'gamma': 1.0, 'gae_lambda': 1.0},
        'request_hash': 'placement-request', 'torch_rng': torch.get_rng_state(),
        'python_rng': ('saved',),
        'progress': {'episodes': 100, 'iteration': 1, 'best_iteration': 1,
                     'best_score': 0.4, 'selection_metric': 'placement_chip_score',
                     'reward_mode': 'game_placement',
                     'validation': {'selection_score': 0.4, 'bb100': 999}},
    }


def checkpoint_bytes(checkpoint):
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


class PPOContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scores_follow_players_across_rotating_seats_and_reset(self):
        context = GameContext()
        hello = {'type': 'hello', 'num_players': 3, 'num_hands': 100, 'stack': 200, 'player': 2}
        context.send(hello)
        first = observation([2, 0, 1], 0)
        encoded = context.encode(first)
        self.assertEqual(encoded[:FEATURE_COUNT], encode_observation(first))
        self.assertEqual(encoded[FEATURE_COUNT:], [0.0] * 7)

        context.send({'type': 'hand_end', 'hand': 0, 'players': [1, 2, 0],
                      'deltas': [30, -50, 20]})
        second = observation([0, 2, 1], 1, hand=1)
        encoded = context.encode(second)
        self.assertEqual(len(encoded), PLACEMENT_FEATURE_COUNT)
        self.assertEqual(encoded[FEATURE_COUNT:], [0.01, -50 / 20000, 30 / 20000,
                                                  20 / 20000, 0.0, 0.0, 0.0])
        context.send(hello)
        self.assertEqual(context.encode(first)[FEATURE_COUNT:], [0.0] * 7)

    def test_future_and_private_fields_do_not_enter_context(self):
        context = GameContext()
        context.send({'type': 'hello', 'num_players': 2, 'num_hands': 100, 'stack': 200, 'player': 0})
        current = observation([0, 1], 0)
        before = context.encode(current)
        context.send({'type': 'street', 'board': ['2c', '3c', '4c']})
        context.send({'type': 'match_end', 'chips': [10000, -10000]})
        enriched = {**current, 'final_chips': [10000, -10000], 'opponent_hole': ['Ah', 'Ad']}
        self.assertEqual(context.encode(enriched), before)
        # Only an actual completed-hand event can advance public scores.
        context.send({'type': 'hand_end', 'hand': 0, 'players': [0, 1], 'deltas': [2, -2],
                      'revealed': {'1': ['Ah', 'Ad']}})
        self.assertEqual(context.encode(observation([1, 0], 1, 1))[FEATURE_COUNT + 1:],
                         [2 / 20000, -2 / 20000, 0.0, 0.0, 0.0, 0.0])

    def test_missing_hello_and_duplicate_results_fail_clearly(self):
        context = GameContext()
        with self.assertRaisesRegex(ValueError, 'hello'):
            context.encode(observation([0, 1], 0))
        context.send({'type': 'hello', 'num_players': 2, 'num_hands': 100, 'stack': 200, 'player': 0})
        result = {'type': 'hand_end', 'hand': 0, 'players': [0, 1], 'deltas': [2, -2]}
        context.send(result)
        with self.assertRaisesRegex(ValueError, 'twice'):
            context.send(result)

    def test_bc_zero_extension_preserves_outputs_and_source_weights(self):
        source = build_model()
        original = deepcopy(source.state_dict())
        actor = build_actor(source.state_dict(), context=True)
        features = torch.randn(16, FEATURE_COUNT)
        context = torch.randn(16, 7)
        extended = torch.cat([features, context], dim=1)
        torch.testing.assert_close(actor(extended), source(features), rtol=1e-6, atol=1e-6)
        self.assertEqual(actor[0].in_features, PLACEMENT_FEATURE_COUNT)
        self.assertEqual(build_critic(context=True)[0].in_features, PLACEMENT_FEATURE_COUNT)
        self.assertEqual(build_actor()[0].in_features, FEATURE_COUNT)
        self.assertEqual(build_critic()[0].in_features, FEATURE_COUNT)
        self.assertEqual(torch.count_nonzero(actor[0].weight[:, FEATURE_COUNT:]).item(), 0)
        for name, weights in source.state_dict().items():
            self.assertTrue(torch.equal(weights, original[name]))
        restored = build_actor(actor.state_dict(), context=True)
        torch.testing.assert_close(restored(extended), actor(extended), rtol=0, atol=0)

    def test_legacy_and_context_exports_have_only_the_required_files(self):
        legacy = cloning.playable_files('Legacy', b'{}')
        model = {'feature_version': FEATURE_VERSION, 'feature_count': PLACEMENT_FEATURE_COUNT,
                 'context_version': CONTEXT_VERSION, 'layers': model_layers(build_actor(context=True))}
        context = cloning.playable_files('EXTRA_IMPORT FEATURE_ENCODING STUDENT_NAME', json.dumps(model).encode())
        self.assertEqual(set(legacy), {'main.py', 'weights.json', 'bc_features.py'})
        self.assertEqual(set(context), {'main.py', 'weights.json', 'bc_features.py', 'ppo_context.py'})
        compile(legacy['main.py'], 'main.py', 'exec')
        compile(context['main.py'], 'main.py', 'exec')
        self.assertNotIn(b'import torch', context['ppo_context.py'])
        self.assertIn(b'on_hand_end', context['main.py'])
        self.assertIn(b'EXTRA_IMPORT FEATURE_ENCODING STUDENT_NAME', context['main.py'])

    def test_snapshot_loader_requires_matching_context_metadata_and_shapes(self):
        original = placement_checkpoint()
        _, progress, layers, is_bc = ppo_snapshot_worker.load_actor(checkpoint_bytes(original))
        self.assertFalse(is_bc)
        self.assertEqual(progress['best_score'], 0.4)
        self.assertEqual(len(layers[0]['weight'][0]), PLACEMENT_FEATURE_COUNT)
        for field, value in [('context_version', None), ('context_version', 2), ('context_version', True),
                             ('feature_count', FEATURE_COUNT), ('reward_mode', 'chips'),
                             ('selection_metric', 'bb100')]:
            changed = deepcopy(original)
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                ppo_snapshot_worker.load_actor(checkpoint_bytes(changed))
        changed = deepcopy(original)
        changed['actor']['0.weight'] = changed['actor']['0.weight'][:, :FEATURE_COUNT]
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            ppo_snapshot_worker.load_actor(checkpoint_bytes(changed))

    def test_context_snapshot_exports_and_plays_real_sdk_games_at_two_through_six_players(self):
        # Keep every temporary export inside this staging test directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            data = Path(temporary)
            job_id = 'c' * 32
            folder = data / 'ppo_runs' / job_id
            folder.mkdir(parents=True)
            checkpoint = placement_checkpoint()
            saved = checkpoint_bytes(checkpoint)
            (folder / 'best.pt').write_bytes(saved)
            job = {'id': job_id, 'name': 'Context SDK check', 'status': 'paused',
                   'config': {'reward_mode': 'game_placement'}}
            server = SimpleNamespace(DATA=data, get=lambda table, key: job)
            result = ppo_snapshots.prepare(server, job_id)
            self.assertEqual(result['selection_metric'], 'placement_chip_score')
            self.assertEqual(result['score'], 0.4)
            self.assertEqual(result['context_version'], CONTEXT_VERSION)
            self.assertEqual(result['feature_count'], PLACEMENT_FEATURE_COUNT)
            self.assertEqual(result['reward_mode'], 'game_placement')
            self.assertEqual([row['table_size'] for row in result['validation']], [2, 3, 4, 5, 6])
            self.assertTrue(all(row['hands'] == 2 and set(row['verdicts']) == {'OK'} for row in result['validation']))
            self.assertEqual((folder / 'best.pt').read_bytes(), saved)
            with zipfile.ZipFile(ppo_snapshots.artifact(server, result['snapshot_id'])) as archive:
                self.assertEqual(set(archive.namelist()), {'main.py', 'weights.json', 'bc_features.py', 'ppo_context.py'})
                weights = json.loads(archive.read('weights.json'))
                self.assertEqual(weights['context_version'], CONTEXT_VERSION)
                self.assertEqual(len(weights['layers'][0]['weight'][0]), PLACEMENT_FEATURE_COUNT)


if __name__ == '__main__':
    unittest.main()
