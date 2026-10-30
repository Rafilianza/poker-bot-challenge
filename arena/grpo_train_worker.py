"""Critic-free GRPO fine-tuning from a BC actor and complete poker games.

Each game gets the same total weight even when its number of decisions differs.
Decision chunks limit memory; they accumulate into one optimizer step per epoch.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import traceback

from arena.bc_features import FEATURE_VERSION
from arena.bc_train_worker import get_device
from arena.ppo_train_worker import TrainingPaused, check_pause, request_pause
from arena.ppo_train_worker import write_json, file_hash, export_weights, masked_log_probabilities


def trajectory_weights(transitions):
    """Give each sampled game one vote, shared across its decisions."""
    counts = {}
    for row in transitions:
        trajectory = row['trajectory_id']
        counts[trajectory] = counts.get(trajectory, 0) + 1
    if not counts:
        return []
    return [1.0 / (len(counts) * counts[row['trajectory_id']]) for row in transitions]


def weighted_policy_loss(new_log_probabilities, old_log_probabilities, advantages, weights, clip_ratio):
    import torch
    log_ratio = new_log_probabilities - old_log_probabilities
    ratios = log_ratio.exp()
    ordinary = ratios * advantages
    clipped = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    loss = -(torch.minimum(ordinary, clipped) * weights).sum()
    approximate_kl = (((ratios - 1.0) - log_ratio) * weights).sum()
    clip_fraction = (((ratios - 1.0).abs() > clip_ratio).float() * weights).sum()
    return loss, approximate_kl, clip_fraction


def update_policy(actor, reference_actor, optimizer, transitions, config, device,
                  completed_episodes, folder):
    """Apply clipped relative rewards without creating or training a critic."""
    import torch
    remaining = max(0.0, 1.0 - completed_episodes / config['target_episodes'])
    anchor_weight = config['bc_coef'] * remaining
    # Identical outcomes cannot tell GRPO which sampled action was better.
    # Skip these groups entirely, including entropy/anchor-only updates.
    informative = [row for row in transitions if row['informative_group']]
    empty_metrics = {'policy_loss': None, 'entropy': None, 'approx_kl': None,
                     'clip_fraction': None, 'bc_kl': None, 'bc_coef': anchor_weight,
                     'updates': 0, 'early_stopped': False, 'update_skipped': True}
    if not informative:
        return empty_metrics
    if any(not math.isfinite(row['advantage']) for row in informative):
        raise ValueError('GRPO advantages must be finite')
    # Keep the rollout on CPU and move only one decision chunk to the device.
    features = torch.tensor([row['features'] for row in informative], dtype=torch.float32)
    masks = torch.tensor([row['mask'] for row in informative], dtype=torch.bool)
    actions = torch.tensor([row['action'] for row in informative], dtype=torch.long)
    old_log_probabilities = torch.tensor([row['old_log_prob'] for row in informative], dtype=torch.float32)
    advantages = torch.tensor([row['advantage'] for row in informative], dtype=torch.float32)
    weights = torch.tensor(trajectory_weights(informative), dtype=torch.float32)
    if not masks.gather(1, actions.unsqueeze(1)).all():
        raise ValueError('A rollout contains an action excluded by its saved legal mask')
    metric_names = ('policy_loss', 'entropy', 'approx_kl', 'clip_fraction', 'bc_kl')

    def measure_epoch(backward):
        totals = {name: 0.0 for name in metric_names}
        for start in range(0, len(informative), config['batch_size']):
            check_pause(folder)
            stop = start + config['batch_size']
            batch_features = features[start:stop].to(device)
            batch_masks = masks[start:stop].to(device)
            batch_weights = weights[start:stop].to(device)
            log_probabilities = masked_log_probabilities(actor(batch_features), batch_masks)
            selected = log_probabilities.gather(1, actions[start:stop].to(device).unsqueeze(1)).squeeze(1)
            loss, kl, clipped = weighted_policy_loss(
                selected, old_log_probabilities[start:stop].to(device),
                advantages[start:stop].to(device), batch_weights, config['clip_ratio'],
            )
            probabilities = log_probabilities.exp()
            entropy = (-(probabilities * log_probabilities).sum(dim=1) * batch_weights).sum()
            with torch.no_grad():
                reference_log = masked_log_probabilities(reference_actor(batch_features), batch_masks)
            bc_kl = ((reference_log.exp() * (reference_log - log_probabilities)).sum(dim=1) * batch_weights).sum()
            actor_loss = loss - config['entropy_coef'] * entropy + anchor_weight * bc_kl
            if not torch.isfinite(actor_loss):
                raise ValueError('GRPO produced a non-finite update')
            if backward:
                actor_loss.backward()
            for name, value in zip(metric_names, (loss, entropy, kl, clipped, bc_kl)):
                totals[name] += float(value.detach().cpu())
        return totals

    actor.train()
    updates = 0
    early_stopped = False
    for update_epoch in range(config['update_epochs']):
        optimizer.zero_grad()
        metrics = measure_epoch(backward=True)
        if updates and metrics['approx_kl'] > config['target_kl']:
            optimizer.zero_grad()
            early_stopped = True
            break
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        optimizer.step()
        updates += 1
    actor.eval()
    with torch.no_grad():
        metrics = measure_epoch(backward=False)
    metrics.update(bc_coef=anchor_weight, updates=updates, early_stopped=early_stopped,
                   update_skipped=False)
    return metrics


def save_checkpoint(path, actor, optimizer, progress, request_hash, device):
    import torch
    state = {'algorithm': 'grpo', 'actor': actor.state_dict(),
             'actor_optimizer': optimizer.state_dict(), 'progress': deepcopy(progress),
             'request_hash': request_hash, 'feature_version': FEATURE_VERSION,
             'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}
    state.update(deepcopy(getattr(actor, '_ppo_metadata', {})))
    if device.type == 'mps':
        state['mps_rng'] = torch.mps.get_rng_state()
    if device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, actor, optimizer, request_hash, device):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('algorithm') != 'grpo' or saved['request_hash'] != request_hash
            or saved['feature_version'] != FEATURE_VERSION):
        raise ValueError('The saved GRPO checkpoint belongs to different training settings')
    for name, expected in getattr(actor, '_ppo_metadata', {}).items():
        if saved.get(name) != expected:
            raise ValueError('The saved GRPO checkpoint has different reward or context settings')
    actor.load_state_dict(saved['actor'])
    optimizer.load_state_dict(saved['actor_optimizer'])
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if torch.is_tensor(value):
                parameter_state[name] = value.to(device)
    torch.set_rng_state(saved['torch_rng'])
    random.setstate(saved['python_rng'])
    if device.type == 'mps' and 'mps_rng' in saved:
        torch.mps.set_rng_state(saved['mps_rng'])
    if device.type == 'cuda' and 'cuda_rng' in saved:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['progress']


def train(request, folder, report):
    import torch
    from arena.grpo_rollout import collect_rollout, validate_groups
    from arena.ppo_rollout import evaluate_policy

    torch.set_num_threads(min(2, torch.get_num_threads()))
    config = request['config']
    reward_mode = 'game_placement'
    selection_key = 'selection_score'
    from arena.ppo_objectives import reward_settings
    reward_settings(config)
    group_size, group_episodes = validate_groups(config, 0, config['rollout_episodes'])
    if (type(config['target_episodes']) is not int or config['target_episodes'] < group_episodes
            or config['target_episodes'] % group_episodes):
        raise ValueError('GRPO target episodes must contain complete groups')
    if config.get('eval_hands', 100) != 100:
        raise ValueError('GRPO evaluation requires complete 100-hand games')
    if request.get('feature_version') != FEATURE_VERSION:
        raise ValueError('The selected checkpoint uses a different observation encoder')
    source_path = Path(request['source_checkpoint']['path'])
    if file_hash(source_path) != request['source_checkpoint']['sha256']:
        raise ValueError('The frozen BC checkpoint checksum changed')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    device = get_device(config['device'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    from arena.ppo_models import build_actor
    from arena.ppo_context import CONTEXT_VERSION, PLACEMENT_FEATURE_COUNT
    actor = build_actor(source['model'], context=True).to(device)
    actor._ppo_metadata = {
        'algorithm': 'grpo', 'feature_count': PLACEMENT_FEATURE_COUNT,
        'context_version': CONTEXT_VERSION, 'reward_mode': reward_mode,
        'selection_metric': 'placement_chip_score', 'reward_config': reward_settings(config),
    }
    reference_actor = deepcopy(actor).eval()
    for parameter in reference_actor.parameters():
        parameter.requires_grad_(False)
    # BC optimizer state is deliberately not reused for the new RL objective.
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config['learning_rate'])
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    checkpoint_path = folder / 'latest.pt'
    progress = {'episodes': 0, 'target_episodes': config['target_episodes'], 'iteration': 0,
                'decisions': 0, 'groups': 0, 'history': [], 'baseline': None, 'validation': None,
                'best_score': None, 'best_validation': None, 'best_iteration': 0}
    progress.update(algorithm='grpo', reward_mode=reward_mode, selection_metric='placement_chip_score')
    if checkpoint_path.exists():
        progress = restore_checkpoint(checkpoint_path, actor, actor_optimizer, request_hash, device)
    else:
        save_checkpoint(checkpoint_path, actor, actor_optimizer, progress, request_hash, device)
    # Milestone settings are immutable and stored separately from the request.
    # Direct worker tests can omit the optional benchmark sidecar.
    milestones = None
    if (folder / 'milestone-settings.json').exists():
        from arena import ppo_milestones
        milestones = ppo_milestones.begin_tracking(folder, request, progress['episodes'])
        if milestones is not None and milestones['settings']['interval'] % group_episodes:
            raise ValueError('GRPO milestone intervals must align with complete groups')
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
        save_checkpoint(folder / 'best.pt', actor, actor_optimizer, progress, request_hash, device)
        save_checkpoint(checkpoint_path, actor, actor_optimizer, progress, request_hash, device)
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
        report.update(phase='Collecting matched groups',
                      progress=f'Iteration {iteration}: playing {episode_count // group_episodes:,} groups ({episode_count:,} hands)')
        report_status()
        rollout = collect_rollout(actor, request['opponents'], config, progress['episodes'],
                                  episode_count, folder / 'rollouts' / f'iteration-{iteration}',
                                  check_pause=pause_check)
        if rollout['episodes'] != episode_count:
            raise ValueError('GRPO collection did not return the requested complete hands')
        report.update(phase='Updating GRPO', progress=f'Iteration {iteration}: comparing grouped game results')
        report_status()
        metrics = update_policy(actor, reference_actor, actor_optimizer,
                                rollout['transitions'], config, device, progress['episodes'], folder)
        episodes = progress['episodes'] + rollout['episodes']
        chip_total = sum(rollout['chips_by_size'].values())
        history_row = {'iteration': iteration, 'episodes': episodes,
                       'train_bb100': chip_total / 2.0 / rollout['episodes'] * 100.0, **metrics}
        history_row.update(train_reward=rollout['train_reward'], groups=rollout['groups'],
                           reward_std=rollout['reward_std'], zero_spread_fraction=rollout['zero_spread_fraction'])
        progress.update(episodes=episodes, iteration=iteration,
                        groups=progress['groups'] + rollout['groups'],
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
            history_row['validation_score'] = validation[selection_key]
            if validation[selection_key] > progress['best_score']:
                progress.update(best_score=validation[selection_key], best_validation=validation, best_iteration=iteration)
                improved = True
        if improved:
            save_checkpoint(folder / 'best.pt', actor, actor_optimizer, progress, request_hash, device)
        save_checkpoint(checkpoint_path, actor, actor_optimizer, progress, request_hash, device)
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
    report.update(deepcopy(progress), status='completed', phase='Completed', progress='GRPO student ready to export',
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
    report = {'status': 'running', 'phase': 'Starting', 'progress': 'Loading GRPO training',
              'episodes': 0, 'iteration': 0, 'decisions': 0, 'groups': 0, 'history': []}
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
        report.update(status='failed', phase='Failed', error=str(error), progress='GRPO training stopped')
        write_json(folder / 'status.json', report)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
