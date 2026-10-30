"""Fine-tune a cloned poker actor with fresh, masked PPO rollouts.

The critic is a separate network. Observations and legal action classes remain
identical to the behaviour-cloning student, so the actor can still be exported
as a small scaffold-compatible bot.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import signal
import time
import traceback

from arena.bc_features import ACTION_NAMES, FEATURE_COUNT, FEATURE_VERSION
from arena.bc_train_worker import build_model, get_device, model_layers

PAUSE_SIGNAL = False


class TrainingPaused(Exception):
    pass


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def request_pause(signum, frame):
    global PAUSE_SIGNAL
    PAUSE_SIGNAL = True


def check_pause(folder):
    if PAUSE_SIGNAL:
        raise TrainingPaused()
    path = folder / 'control.json'
    if path.exists() and json.loads(path.read_text()).get('action') == 'pause':
        raise TrainingPaused()


def build_critic():
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_COUNT, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )


def calculate_advantages(rewards, values, dones, gamma=1.0, gae_lambda=0.95):
    """Compute GAE without crossing a trajectory's terminal boundary.

    Legacy trajectories end per hand; placement trajectories end per game.
    Gamma and lambda both equal one for complete-game Monte Carlo returns.
    """
    if not rewards or not (len(rewards) == len(values) == len(dones)):
        raise ValueError('PPO requires matching, non-empty rewards, values and terminal flags')
    if not dones[-1]:
        raise ValueError('Collect complete hands or games before a PPO update')
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if dones[index]:
            next_value = 0.0
            next_advantage = 0.0
        else:
            next_value = values[index + 1]
        delta = rewards[index] + gamma * next_value - values[index]
        advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = advantage
        next_advantage = advantage
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def masked_log_probabilities(logits, masks):
    import torch
    if not masks.any(dim=-1).all():
        raise ValueError('Every PPO observation must have a legal action')
    return torch.nn.functional.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)


def clipped_policy_loss(new_log_probabilities, old_log_probabilities, advantages, clip_ratio):
    """Use the smaller clipped/unclipped improvement for either advantage sign."""
    import torch
    log_ratio = new_log_probabilities - old_log_probabilities
    ratios = log_ratio.exp()
    ordinary = ratios * advantages
    clipped = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    loss = -torch.minimum(ordinary, clipped).mean()
    approximate_kl = ((ratios - 1.0) - log_ratio).mean()
    clip_fraction = ((ratios - 1.0).abs() > clip_ratio).float().mean()
    return loss, approximate_kl, clip_fraction


def update_policy(actor, critic, reference_actor, actor_optimizer, critic_optimizer,
                  transitions, config, device, completed_episodes, folder):
    """Make a few bounded passes over one fresh rollout, then discard it."""
    import torch
    if not transitions:
        raise ValueError('This rollout contained no learner decisions. Increase rollout episodes.')
    if config.get('reward_mode') == 'game_placement' and (config['gamma'] != 1 or config['gae_lambda'] != 1):
        raise ValueError('Placement PPO requires gamma=1 and gae_lambda=1 for complete-game credit')
    features = torch.tensor([row['features'] for row in transitions], dtype=torch.float32, device=device)
    masks = torch.tensor([row['mask'] for row in transitions], dtype=torch.bool, device=device)
    actions = torch.tensor([row['action'] for row in transitions], dtype=torch.long, device=device)
    old_log_probabilities = torch.tensor([row['old_log_prob'] for row in transitions],
                                         dtype=torch.float32, device=device)
    if not masks.gather(1, actions.unsqueeze(1)).all():
        raise ValueError('A rollout contains an action excluded by its saved legal mask')
    advantages, returns = calculate_advantages(
        [row['reward'] for row in transitions], [row['value'] for row in transitions],
        [row['done'] for row in transitions], config['gamma'], config['gae_lambda'],
    )
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    # The frozen BC policy is an early guide; its influence decreases as fresh
    # experience accumulates. It is not an extra source of reward.
    remaining = max(0.0, 1.0 - completed_episodes / config['target_episodes'])
    anchor_weight = config['bc_coef'] * remaining
    with torch.no_grad():
        reference_log_probabilities = masked_log_probabilities(reference_actor(features), masks)
        reference_probabilities = reference_log_probabilities.exp()
    actor.train()
    critic.train()
    totals = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
              'approx_kl': 0.0, 'clip_fraction': 0.0, 'bc_kl': 0.0}
    examples_updated = 0
    updates = 0
    early_stopped = False
    for update_epoch in range(config['update_epochs']):
        order = torch.randperm(len(transitions))
        for start in range(0, len(order), config['batch_size']):
            check_pause(folder)
            indices = order[start:start + config['batch_size']].to(device)
            batch_masks = masks[indices]
            log_probabilities = masked_log_probabilities(actor(features[indices]), batch_masks)
            selected_log_probabilities = log_probabilities.gather(1, actions[indices].unsqueeze(1)).squeeze(1)
            policy_loss, approximate_kl, clip_fraction = clipped_policy_loss(
                selected_log_probabilities, old_log_probabilities[indices], advantages[indices], config['clip_ratio'],
            )
            if updates and approximate_kl.detach().item() > config['target_kl']:
                early_stopped = True
                break
            probabilities = log_probabilities.exp()
            entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
            bc_kl = (reference_probabilities[indices] * (
                reference_log_probabilities[indices] - log_probabilities)).sum(dim=1).mean()
            actor_loss = policy_loss - config['entropy_coef'] * entropy + anchor_weight * bc_kl
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()
            values = critic(features[indices]).squeeze(-1)
            value_loss = torch.nn.functional.mse_loss(values, returns[indices])
            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
            batch_count = len(indices)
            for name, value in (('policy_loss', policy_loss), ('value_loss', value_loss),
                                ('entropy', entropy), ('approx_kl', approximate_kl),
                                ('clip_fraction', clip_fraction), ('bc_kl', bc_kl)):
                totals[name] += float(value.detach().cpu()) * batch_count
            examples_updated += batch_count
            updates += 1
        if early_stopped:
            break
    if not examples_updated:
        raise ValueError('PPO did not perform an update')
    metrics = {name: total / examples_updated for name, total in totals.items()}
    # Report the resulting policy's KL, not only its values before each step.
    final_kl = 0.0
    final_clip_fraction = 0.0
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(transitions), config['batch_size']):
            check_pause(folder)
            stop = start + config['batch_size']
            log_probabilities = masked_log_probabilities(actor(features[start:stop]), masks[start:stop])
            selected = log_probabilities.gather(1, actions[start:stop].unsqueeze(1)).squeeze(1)
            _, approximate_kl, clip_fraction = clipped_policy_loss(
                selected, old_log_probabilities[start:stop], advantages[start:stop], config['clip_ratio'],
            )
            count = len(selected)
            final_kl += approximate_kl.item() * count
            final_clip_fraction += clip_fraction.item() * count
    metrics.update(approx_kl=final_kl / len(transitions),
                   clip_fraction=final_clip_fraction / len(transitions),
                   bc_coef=anchor_weight, updates=updates, early_stopped=early_stopped)
    return metrics


def actor_state_hash(actor_state):
    """Fingerprint a frozen opponent without depending on checkpoint file bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(actor_state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def opponent_version(actor, iteration):
    actor_state = {name: value.detach().cpu().clone()
                   for name, value in actor.state_dict().items()}
    return {'iteration': iteration, 'actor': actor_state,
            'sha256': actor_state_hash(actor_state)}


def save_opponent_version(self_play_state, snapshot, history_limit):
    """Keep the original BC policy and a small window of older committed policies."""
    history = self_play_state['history']
    if any(saved['iteration'] == snapshot['iteration'] for saved in history):
        return
    history.append(snapshot)
    # Iteration zero stays available even when later versions roll out of the pool.
    if len(history) > history_limit + 1:
        self_play_state['history'] = [history[0]] + history[-history_limit:]


def available_opponent_versions(self_play_state, iteration):
    if self_play_state is None:
        return []
    return [snapshot for snapshot in self_play_state['history']
            if snapshot['iteration'] < iteration]


def frozen_opponent_history(self_play_state, actor, iteration):
    """Materialize independent frozen actors only for genuinely older versions."""
    opponents = []
    for snapshot in available_opponent_versions(self_play_state, iteration):
        frozen_actor = deepcopy(actor).cpu().eval()
        frozen_actor.load_state_dict(snapshot['actor'])
        for parameter in frozen_actor.parameters():
            parameter.requires_grad_(False)
        opponents.append({'iteration': snapshot['iteration'], 'actor': frozen_actor})
    return opponents


def validate_self_play_state(saved, expected_mode, iteration):
    if not isinstance(saved, dict) or saved.get('version') != 1:
        raise ValueError('The saved PPO checkpoint is missing its self-play state')
    if saved.get('training_mode') != expected_mode:
        raise ValueError('The saved PPO checkpoint has a different self-play mode')
    history = saved.get('history')
    if not isinstance(history, list):
        raise ValueError('The saved self-play history is invalid')
    if expected_mode == 'self_play' and history:
        raise ValueError('Pure self-play must not contain historical opponents')
    if expected_mode == 'mixed' and (not history or history[0].get('iteration') != 0):
        raise ValueError('Mixed self-play is missing its original BC opponent')
    previous_iteration = -1
    for snapshot in history:
        version = snapshot.get('iteration')
        if type(version) is not int or not previous_iteration < version <= iteration:
            raise ValueError('The saved self-play history has an invalid iteration')
        if actor_state_hash(snapshot['actor']) != snapshot.get('sha256'):
            raise ValueError('The saved self-play opponent checksum changed')
        previous_iteration = version


def save_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer, progress,
                    request_hash, device, self_play_state=None):
    import torch
    state = {'actor': actor.state_dict(), 'critic': critic.state_dict(),
             'actor_optimizer': actor_optimizer.state_dict(),
             'critic_optimizer': critic_optimizer.state_dict(), 'progress': deepcopy(progress),
             'request_hash': request_hash, 'feature_version': FEATURE_VERSION,
             'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}
    state.update(deepcopy(getattr(actor, '_ppo_metadata', {})))
    if self_play_state is not None:
        # Opponent tensors belong only in resumable checkpoints, never model metadata.
        state['self_play_state'] = self_play_state
    if device.type == 'mps':
        state['mps_rng'] = torch.mps.get_rng_state()
    if device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer,
                       request_hash, device, self_play_state=None):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['request_hash'] != request_hash or checkpoint['feature_version'] != FEATURE_VERSION:
        raise ValueError('The saved PPO checkpoint belongs to different training settings')
    for name, expected in getattr(actor, '_ppo_metadata', {}).items():
        if checkpoint.get(name) != expected:
            raise ValueError('The saved PPO checkpoint has different reward or context settings')
    if self_play_state is not None:
        saved_self_play = checkpoint.get('self_play_state')
        validate_self_play_state(saved_self_play, self_play_state['training_mode'],
                                 checkpoint['progress']['iteration'])
        self_play_state.clear()
        self_play_state.update(saved_self_play)
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    for optimizer in (actor_optimizer, critic_optimizer):
        for parameter_state in optimizer.state.values():
            for name, value in parameter_state.items():
                if torch.is_tensor(value):
                    parameter_state[name] = value.to(device)
    torch.set_rng_state(checkpoint['torch_rng'])
    random.setstate(checkpoint['python_rng'])
    if device.type == 'mps' and 'mps_rng' in checkpoint:
        torch.mps.set_rng_state(checkpoint['mps_rng'])
    if device.type == 'cuda' and 'cuda_rng' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['progress']


def export_weights(folder, actor, best_iteration):
    weights = {'schema_version': 1, 'feature_version': FEATURE_VERSION,
               'feature_count': FEATURE_COUNT, 'action_names': ACTION_NAMES,
               'layers': model_layers(actor), 'best_iteration': best_iteration}
    weights.update(deepcopy(getattr(actor, '_ppo_metadata', {})))
    write_json(folder / 'weights.json', weights)


def train(request, folder, report):
    import torch
    from arena.ppo_rollout import collect_rollout, evaluate_policy

    torch.set_num_threads(min(2, torch.get_num_threads()))
    config = request['config']
    training_mode = config.get('training_mode', 'bots')
    if training_mode not in ('bots', 'self_play', 'mixed'):
        raise ValueError('Unknown PPO training mode')
    history_limit = config.get('self_play_history_limit', 5)
    history_interval = config.get('self_play_history_interval', 5)
    if training_mode == 'mixed':
        if type(history_limit) is not int or not 1 <= history_limit <= 20:
            raise ValueError('Self-play history limit must be between 1 and 20')
        if type(history_interval) is not int or not 1 <= history_interval <= 1000:
            raise ValueError('Self-play history interval must be between 1 and 1000')
    reward_mode = config.get('reward_mode', 'chips')
    if reward_mode not in ('chips', 'game_placement'):
        raise ValueError('Unknown PPO reward mode')
    placement_mode = reward_mode == 'game_placement'
    selection_key = 'selection_score' if placement_mode else 'bb100'
    if config['gamma'] != 1.0:
        raise ValueError('This poker PPO worker uses gamma=1 for complete hand or game returns')
    if placement_mode:
        from arena.ppo_objectives import reward_settings
        reward_settings(config)
        if config['gae_lambda'] != 1.0:
            raise ValueError('Game placement requires gae_lambda=1 for complete-game returns')
        for key in ('target_episodes', 'rollout_episodes'):
            if type(config[key]) is not int or config[key] < 100 or config[key] % 100:
                raise ValueError(f'Game placement {key} must be a positive multiple of 100 hands')
        if config.get('eval_hands', 100) != 100:
            raise ValueError('Game placement evaluation requires 100-hand games')
    if request.get('feature_version') != FEATURE_VERSION:
        raise ValueError('The selected checkpoint uses a different observation encoder')
    source_path = Path(request['source_checkpoint']['path'])
    if file_hash(source_path) != request['source_checkpoint']['sha256']:
        raise ValueError('The frozen BC checkpoint checksum changed')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    device = get_device(config['device'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if placement_mode:
        from arena.ppo_models import build_actor, build_critic as build_context_critic
        from arena.ppo_context import CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT
        actor = build_actor(source['model'], context=True).to(device)
        critic = build_context_critic(context=True).to(device)
        actor._ppo_metadata = {
            'feature_count': PLACEMENT_FEATURE_COUNT, 'context_version': CONTEXT_VERSION,
            'reward_mode': reward_mode, 'selection_metric': 'placement_chip_score',
            'reward_config': {'reward_mode': reward_mode,
                              'placement_weight': config.get('placement_weight', 0.9),
                              'chip_scale_bb100': config.get('chip_scale_bb100', 100),
                              'gamma': 1.0, 'gae_lambda': 1.0},
        }
    else:
        actor = build_model().to(device)
        actor.load_state_dict(source['model'])
        critic = build_critic().to(device)
    reference_actor = deepcopy(actor).eval()
    for parameter in reference_actor.parameters():
        parameter.requires_grad_(False)
    # BC optimizer state is deliberately not reused for the new RL objective.
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config['learning_rate'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config['critic_learning_rate'])
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    checkpoint_path = folder / 'latest.pt'
    progress = {'episodes': 0, 'target_episodes': config['target_episodes'], 'iteration': 0,
                'decisions': 0, 'history': [], 'baseline': None, 'validation': None,
                'best_score': None, 'best_validation': None, 'best_iteration': 0}
    self_play_state = None
    if training_mode != 'bots':
        self_play_state = {'version': 1, 'training_mode': training_mode, 'history': []}
        if training_mode == 'mixed':
            save_opponent_version(self_play_state, opponent_version(actor, 0), history_limit)
        progress.update(training_mode=training_mode, self_play={
            'mode': training_mode, 'current_iteration': 0, 'games': 0,
            'seat_counts': {'current': 0, 'historical': 0, 'bots': 0},
            'history_iterations_used': [], 'history_versions_available': 0,
            'history_iterations_available': [],
        })
    if placement_mode:
        progress.update(reward_mode=reward_mode, selection_metric='placement_chip_score')
    if checkpoint_path.exists():
        progress = restore_checkpoint(checkpoint_path, actor, critic, actor_optimizer,
                                      critic_optimizer, request_hash, device, self_play_state)
        if training_mode == 'mixed' and len(self_play_state['history']) > history_limit + 1:
            raise ValueError('The saved self-play history exceeds the requested pool size')
    else:
        save_checkpoint(checkpoint_path, actor, critic, actor_optimizer,
                        critic_optimizer, progress, request_hash, device, self_play_state)
    # The optional sidecar preserves the exact request hash of older jobs.
    # Direct worker callers without one retain the original training behavior.
    milestones = None
    if (folder / 'milestone-settings.json').exists():
        from arena import ppo_milestones
        milestones = ppo_milestones.begin_tracking(folder, request, progress['episodes'])
        if placement_mode and milestones is not None and milestones['settings']['interval'] % 100:
            raise ValueError('Placement milestone intervals must align with complete 100-hand games')
    report.update(deepcopy(progress), device=str(device), status='running', error=None)
    timing_path = folder / 'timing.json'
    previous_elapsed = json.loads(timing_path.read_text())['elapsed_seconds'] if timing_path.exists() else 0.0
    session_started = time.monotonic()
    last_report = 0.0

    def report_status():
        nonlocal last_report
        last_report = time.monotonic()
        report['elapsed_seconds'] = previous_elapsed + last_report - session_started
        write_json(timing_path, {'elapsed_seconds': report['elapsed_seconds']})
        write_json(folder / 'status.json', report)

    def pause_check():
        check_pause(folder)
        if time.monotonic() - last_report >= 1.0:
            report_status()

    def benchmark_saved_checkpoint():
        if milestones is None:
            return
        report.update(phase='Benchmarking saved snapshots',
                      progress='Comparing archived policies across fixed 4, 5 and 6-player full games')
        report_status()

        def benchmark_progress(benchmark):
            report.update(milestones=ppo_milestones.summaries(folder),
                          progress=f"{benchmark['snapshot_id']}: {benchmark['attempted_games']}/{benchmark['expected_games']} benchmark games recorded")
            report_status()

        report['milestones'] = ppo_milestones.checkpoint_hook(
            folder, request, progress, milestones, pause_check, benchmark_progress,
        )
        report_status()
    if progress['baseline'] is None:
        report.update(phase='Evaluating BC baseline', progress='Checking the initial cloned actor on fixed validation seeds')
        report_status()
        baseline = evaluate_policy(actor, request['opponents'], config, folder / 'baseline',
                                   split='validation', check_pause=pause_check)
        progress.update(baseline=baseline, validation=baseline, best_validation=baseline, best_score=baseline[selection_key])
        save_checkpoint(folder / 'best.pt', actor, critic, actor_optimizer,
                        critic_optimizer, progress, request_hash, device, self_play_state)
        save_checkpoint(checkpoint_path, actor, critic, actor_optimizer,
                        critic_optimizer, progress, request_hash, device, self_play_state)
        report.update(deepcopy(progress))
        report_status()
    # latest.pt already contains the complete optimizer and RNG states. A
    # resumed benchmark therefore finishes before collecting any new hands.
    benchmark_saved_checkpoint()
    while progress['episodes'] < config['target_episodes']:
        check_pause(folder)
        iteration = progress['iteration'] + 1
        episode_count = min(config['rollout_episodes'], config['target_episodes'] - progress['episodes'])
        if milestones is not None:
            episode_count = ppo_milestones.next_rollout_count(
                progress['episodes'], config['target_episodes'], episode_count,
                milestones['settings']['interval'],
            )
        report.update(phase='Collecting fresh hands', progress=f'Iteration {iteration}: playing {episode_count:,} new hands')
        report_status()
        rollout_config = config
        rollout_options = {}
        pending_opponent_version = None
        if self_play_state is not None:
            # This runtime label is deliberately kept out of the immutable request.
            rollout_config = dict(config, self_play_iteration=progress['iteration'])
            rollout_options['self_play_history'] = frozen_opponent_history(
                self_play_state, actor, progress['iteration'],
            )
            if (training_mode == 'mixed' and progress['iteration'] > 0
                    and progress['iteration'] % history_interval == 0):
                # The current actor is already stored in latest.pt. Add it to
                # history only once a newer actor commits, so even a one-version
                # pool retains a genuinely older policy instead of the current one.
                pending_opponent_version = opponent_version(actor, progress['iteration'])
        rollout = collect_rollout(actor, critic, request['opponents'], rollout_config, progress['episodes'],
                                  episode_count, folder / 'rollouts' / f'iteration-{iteration}',
                                  check_pause=pause_check, **rollout_options)
        if rollout['episodes'] != episode_count:
            raise ValueError('PPO collection did not return the requested complete hands')
        report.update(phase='Updating PPO', progress=f'Iteration {iteration}: updating from {len(rollout["transitions"]):,} learner decisions')
        report_status()
        if rollout['transitions']:
            metrics = update_policy(actor, critic, reference_actor, actor_optimizer, critic_optimizer,
                                    rollout['transitions'], config, device, progress['episodes'], folder)
        else:
            # Other players can fold before the learner acts. Those hands count
            # as episodes, but provide no sampled action to update with PPO.
            metrics = {'policy_loss': None, 'value_loss': None, 'entropy': None,
                       'approx_kl': None, 'clip_fraction': None, 'bc_kl': None,
                       'bc_coef': 0.0, 'updates': 0, 'early_stopped': False}
        episodes = progress['episodes'] + rollout['episodes']
        chip_total = sum(rollout['chips_by_size'].values())
        history_row = {'iteration': iteration, 'episodes': episodes,
                       'train_bb100': chip_total / 2.0 / rollout['episodes'] * 100.0, **metrics}
        if placement_mode:
            history_row['train_reward'] = rollout['train_reward']
        if self_play_state is not None:
            history_row['self_play'] = deepcopy(rollout['self_play'])
            available = available_opponent_versions(self_play_state, progress['iteration'])
            history_row['self_play'].update(
                history_versions_available=len(available),
                history_iterations_available=[snapshot['iteration'] for snapshot in available],
            )
        progress.update(episodes=episodes, iteration=iteration,
                        decisions=progress['decisions'] + len(rollout['transitions']))
        progress['history'].append(history_row)
        should_evaluate = iteration % config['eval_interval'] == 0 or episodes >= config['target_episodes']
        improved = False
        if should_evaluate:
            report.update(phase='Evaluating candidate', progress=f'Iteration {iteration}: checking fixed validation seeds')
            report_status()
            validation = evaluate_policy(actor, request['opponents'], config,
                                         folder / 'validation' / f'iteration-{iteration}',
                                         split='validation', check_pause=pause_check)
            progress['validation'] = validation
            history_row['validation_bb100'] = validation['bb100']
            if placement_mode:
                history_row['validation_score'] = validation[selection_key]
            if validation[selection_key] > progress['best_score']:
                progress.update(best_score=validation[selection_key], best_validation=validation, best_iteration=iteration)
                improved = True
        if self_play_state is not None:
            # Commit the refreshed pool with the same actor/optimizer iteration.
            # An interrupted iteration never adds a version to the saved timeline.
            if pending_opponent_version is not None:
                save_opponent_version(self_play_state, pending_opponent_version, history_limit)
            progress['self_play'] = deepcopy(history_row['self_play'])
            available = available_opponent_versions(self_play_state, iteration)
            progress['self_play'].update(
                history_versions_available=len(available),
                history_iterations_available=[snapshot['iteration'] for snapshot in available],
            )
        if improved:
            save_checkpoint(folder / 'best.pt', actor, critic, actor_optimizer,
                            critic_optimizer, progress, request_hash, device, self_play_state)
        save_checkpoint(checkpoint_path, actor, critic, actor_optimizer,
                        critic_optimizer, progress, request_hash, device, self_play_state)
        report.update(deepcopy(progress), phase='Iteration saved', progress=f'Completed {episodes:,}/{config["target_episodes"]:,} training hands')
        report_status()
        if milestones is not None and (episodes % milestones['settings']['interval'] == 0
                                       or episodes == config['target_episodes']):
            benchmark_saved_checkpoint()
    check_pause(folder)
    best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    actor.load_state_dict(best['actor'])
    report.update(phase='Evaluating held-out test', progress='Testing the validation-selected actor on fresh test seeds')
    report_status()
    test = evaluate_policy(actor, request['opponents'], config, folder / 'test',
                           split='test', check_pause=pause_check)
    report.update(phase='Comparing BC on held-out test', progress='Checking the original BC actor on the same test seeds')
    report_status()
    baseline_test = evaluate_policy(reference_actor, request['opponents'], config, folder / 'baseline-test',
                                    split='test', check_pause=pause_check)
    export_weights(folder, actor, progress['best_iteration'])
    report.update(deepcopy(progress), status='completed', phase='Completed', progress='PPO student ready to export',
                  validation=best['progress']['validation'], best_validation=best['progress']['validation'],
                  latest_validation=progress['validation'], test=test, baseline_test=baseline_test)
    report_status()
    write_json(folder / 'result.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    request_path = Path(args.request)
    folder = request_path.parent
    report = {'status': 'running', 'phase': 'Starting', 'progress': 'Loading PPO training',
              'episodes': 0, 'iteration': 0, 'decisions': 0, 'history': []}
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    try:
        request = json.loads(request_path.read_text())
        report['target_episodes'] = request['config']['target_episodes']
        write_json(folder / 'status.json', report)
        train(request, folder, report)
    except TrainingPaused:
        # Reload the completed iteration's counters. Updates made during a
        # partial iteration are discarded and replayed when the user resumes.
        checkpoint_path = folder / 'latest.pt'
        if checkpoint_path.exists():
            import torch
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            report.update(saved['progress'])
        report.update(status='paused', phase='Paused',
                      progress='Completed iterations saved. Resume replays any unfinished iteration.')
        write_json(folder / 'status.json', report)
    except Exception as error:
        report.update(status='failed', phase='Failed', error=str(error), progress='PPO training stopped')
        write_json(folder / 'status.json', report)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
