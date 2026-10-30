"""CPU-only validation for immutable PPO downloads; never runs training."""
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

from arena import cloning
from arena.bc_features import ACTION_NAMES, FEATURE_COUNT, FEATURE_VERSION
from arena.bc_features import encode_observation, forward_logits
from arena.bc_train_worker import model_layers
from arena.ppo_context import CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT, GameContext
from arena.ppo_models import build_actor


def load_actor(checkpoint_bytes, source_metadata=None):
    """Validate saved dimensions and values before generating inference weights."""
    import torch
    try:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    except Exception as error:
        raise ValueError('The saved checkpoint is corrupt or is not a PyTorch checkpoint') from error
    if not isinstance(checkpoint, dict):
        raise ValueError('The saved checkpoint must contain a model dictionary')
    is_bc = bool(source_metadata and source_metadata.get('checkpoint_format') == 'bc')
    if is_bc:
        version = source_metadata.get('feature_version')
        actor_state = checkpoint.get('model')
        progress = {'episodes': 0, 'iteration': 0, 'best_iteration': 0}
    else:
        version = checkpoint.get('feature_version')
        actor_state = checkpoint.get('actor')
        progress = checkpoint.get('progress')
    if type(version) is not int or version != FEATURE_VERSION:
        raise ValueError('The saved checkpoint uses an incompatible feature version')
    input_count = checkpoint.get('feature_count', FEATURE_COUNT)
    context_version = checkpoint.get('context_version')
    context = input_count == PLACEMENT_FEATURE_COUNT and type(context_version) is int and context_version == CONTEXT_VERSION
    if type(input_count) is not int or (not context and (input_count != FEATURE_COUNT or context_version is not None)):
        raise ValueError('The saved checkpoint uses incompatible feature count or context metadata')
    if is_bc and context:
        raise ValueError('A BC source checkpoint must use the original 189 features')
    if source_metadata:
        if source_metadata.get('feature_count', input_count) != input_count:
            raise ValueError('The milestone and checkpoint feature counts disagree')
        if source_metadata.get('context_version', context_version) != context_version:
            raise ValueError('The milestone and checkpoint context versions disagree')
    if not isinstance(actor_state, dict) or not isinstance(progress, dict):
        raise ValueError('The saved checkpoint is missing its actor or saved progress')
    for counter in ('episodes', 'iteration'):
        if type(progress.get(counter)) is not int or progress[counter] < 0:
            raise ValueError('The checkpoint has invalid saved episode or iteration counters')
    best_iteration = progress.get('best_iteration', progress['iteration'])
    if type(best_iteration) is not int or not 0 <= best_iteration <= progress['iteration']:
        raise ValueError('The checkpoint has an invalid best iteration')
    score = progress.get('best_score')
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
        raise ValueError('The checkpoint has a non-finite validation score')
    reward_mode = checkpoint.get('reward_mode', 'chips')
    if context:
        from arena.ppo_objectives import reward_settings
        reward_config = checkpoint.get('reward_config')
        if reward_mode != 'game_placement' or not isinstance(reward_config, dict):
            raise ValueError('A context checkpoint needs saved game-placement reward settings')
        if reward_settings(reward_config)['reward_mode'] != reward_mode:
            raise ValueError('The checkpoint reward settings disagree with its reward mode')
        if checkpoint.get('selection_metric') != 'placement_chip_score':
            raise ValueError('A placement checkpoint needs its saved selection metric')
        if progress.get('selection_metric', checkpoint['selection_metric']) != checkpoint['selection_metric']:
            raise ValueError('The checkpoint and saved progress selection metrics disagree')
        if progress.get('reward_mode', reward_mode) != reward_mode:
            raise ValueError('The checkpoint and saved progress reward modes disagree')
    elif reward_mode != 'chips':
        raise ValueError('Placement reward requires the public game-context encoder')

    actor = build_actor(context=context).cpu()
    expected = actor.state_dict()
    if set(actor_state) != set(expected):
        raise ValueError('The saved actor has incompatible layer names')
    for name, tensor in actor_state.items():
        if not torch.is_tensor(tensor) or tensor.shape != expected[name].shape:
            raise ValueError('The saved actor has incompatible layer dimensions')
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise ValueError('The saved actor contains non-finite or invalid weights')
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    layers = model_layers(actor)
    # Exercise the very same standard-library encoder and forward pass as the ZIP.
    observation = {
        'seat': 0, 'street': 'preflop', 'hole': ['As', 'Kd'], 'board': [],
        'pot': 3, 'to_call': 2, 'min_raise_to': 4, 'max_raise_to': 200,
        'can_raise': True, 'stacks': [200, 199, 198, 200],
        'street_bets': [0, 1, 2, 0], 'folded': [False] * 4,
        'button': 0, 'history': [], 'hand': 1, 'players': [0, 1, 2, 3],
    }
    if context:
        game_context = GameContext()
        game_context.send({'type': 'hello', 'num_players': 4, 'num_hands': 100,
                           'stack': 200, 'player': 0})
        game_context.send({'type': 'hand_end', 'hand': 0, 'players': [2, 3, 0, 1],
                           'deltas': [12, -2, -4, -6]})
        features = game_context.encode(observation)
    else:
        features = encode_observation(observation)
    with torch.no_grad():
        torch_logits = actor(torch.tensor(features, dtype=torch.float32))
    python_logits = forward_logits(features, layers)
    if not torch.isfinite(torch_logits).all().item() or not all(map(math.isfinite, python_logits)):
        raise ValueError('The saved actor produces non-finite inference output')
    if not torch.allclose(torch_logits, torch.tensor(python_logits), rtol=1e-4, atol=1e-5):
        raise ValueError('The exported actor does not match PyTorch inference')
    return checkpoint, progress, layers, is_bc


def validate_games(folder):
    validation = []
    game_worker = Path(__file__).with_name('game_worker.py')
    for table_size in (2, 3, 4, 5, 6):
        request_path = folder / f'check-{table_size}.request.json'
        result_path = folder / f'check-{table_size}.json'
        request = {
            'specs': [str(folder / 'bot' / 'main.py')] + ['house:call'] * (table_size - 1),
            'config': {'seats': table_size, 'deals': 2, 'seed': 'snapshot-legal-check',
                       'base_time_ms': 10000, 'increment_ms': 100},
        }
        cloning.write_json(request_path, request)
        process = subprocess.run([sys.executable, str(game_worker), str(request_path), str(result_path)],
                                 capture_output=True, timeout=25)
        if process.returncode or not result_path.is_file():
            raise ValueError('SDK validation failed: ' + process.stderr.decode(errors='replace')[-1000:])
        result = json.loads(result_path.read_text())
        verdicts = result.get('verdicts', [])
        if len(verdicts) != table_size or any(verdict != 'OK' for verdict in verdicts):
            raise ValueError('The saved actor failed SDK legal-action validation: ' + str(verdicts))
        if result.get('num_hands', 0) < 1:
            raise ValueError('SDK validation did not play a hand')
        validation.append({'table_size': table_size, 'hands': result['num_hands'], 'verdicts': verdicts})
    return validation


def export(folder):
    request = json.loads((folder / 'export-request.json').read_text())
    identity = request['identity']
    checkpoint_bytes = (folder / 'full.pt').read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != identity['checkpoint_sha256']:
        raise ValueError('The copied checkpoint checksum changed')
    checkpoint, progress, layers, is_bc = load_actor(checkpoint_bytes, request.get('source_metadata'))
    validation_score = progress.get('best_score') if identity['kind'] == 'best' else None
    training_metric = progress.get('selection_metric', checkpoint.get('selection_metric', 'bb100'))
    selection_metric = training_metric if identity['kind'] == 'best' else 'episode_archive'
    input_count = checkpoint.get('feature_count', FEATURE_COUNT)
    context_version = checkpoint.get('context_version')
    required_resume_fields = {'actor', 'critic', 'actor_optimizer', 'critic_optimizer',
                              'progress', 'request_hash', 'feature_version', 'torch_rng', 'python_rng'}
    metadata = {
        'episodes': progress['episodes'], 'iteration': progress['iteration'],
        'best_iteration': progress.get('best_iteration', progress['iteration']),
        'score': validation_score, 'selection_metric': selection_metric,
        'validation': progress.get('validation'), 'feature_version': FEATURE_VERSION,
        'feature_count': input_count, 'checkpoint_format': 'bc' if is_bc else 'ppo',
        'reward_mode': checkpoint.get('reward_mode', 'chips'),
        'reward_config': checkpoint.get('reward_config'), 'training_selection_metric': training_metric,
        'resumable': not is_bc and required_resume_fields.issubset(checkpoint),
        'request_hash': checkpoint.get('request_hash'),
        'device': 'cpu',
    }
    weights = {'schema_version': 1, 'feature_version': FEATURE_VERSION,
               'feature_count': input_count, 'action_names': ACTION_NAMES,
               'layers': layers, 'best_iteration': metadata['best_iteration']}
    if context_version is not None:
        metadata['context_version'] = context_version
        weights.update(context_version=context_version, reward_mode=metadata['reward_mode'],
                       reward_config=metadata['reward_config'], selection_metric=training_metric)
    weights_bytes = json.dumps(weights, allow_nan=False).encode()
    files = cloning.playable_files(identity['name'], weights_bytes)
    for filename, hash_key in (('main.py', 'main_sha256'), ('bc_features.py', 'encoder_sha256')):
        if hashlib.sha256(files[filename]).hexdigest() != identity[hash_key]:
            raise ValueError('The export code changed while preparing the snapshot; retry')
    if context_version is not None:
        if hashlib.sha256(files['ppo_context.py']).hexdigest() != identity.get('context_sha256'):
            raise ValueError('The game-context encoder changed while preparing the snapshot; retry')
    bot_folder = folder / 'bot'
    bot_folder.mkdir()
    for filename, content in files.items():
        (bot_folder / filename).write_bytes(content)
    validation = validate_games(folder)
    # Legacy agents have three files; placement agents add only the shared
    # standard-library context module, never training dependencies.
    with zipfile.ZipFile(folder / 'bot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            item = zipfile.ZipInfo(filename, date_time=(2020, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, content)
    report = {**identity, **metadata, 'metadata': metadata, 'validation': validation,
              'zip_available': True, 'checkpoint_available': True,
              'zip_sha256': hashlib.sha256((folder / 'bot.zip').read_bytes()).hexdigest()}
    (folder / 'report.json').write_text(json.dumps(report, allow_nan=False))
    return report


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        export(Path(sys.argv[1]).resolve())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
