"""Collect matched 100-hand game groups for critic-free policy learning."""
import math

from arena.ppo_rollout import frozen_cpu_model, play_game, rollout_settings, selected_opponents


def group_advantages(rewards):
    """Compare complete-game rewards only with repetitions of the same game."""
    if len(rewards) < 2 or not all(math.isfinite(reward) for reward in rewards):
        raise ValueError('A GRPO group needs at least two finite rewards')
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    reward_std = math.sqrt(variance)
    if reward_std <= 1e-8:
        return [0.0] * len(rewards), reward_std
    return [(reward - mean_reward) / reward_std for reward in rewards], reward_std


def validate_groups(config, episode_start, episode_count):
    group_size = config.get('group_size', 4)
    if type(group_size) is not int or not 2 <= group_size <= 8:
        raise ValueError('GRPO group size must be an integer from 2 to 8')
    if config.get('reward_mode') != 'game_placement':
        raise ValueError('GRPO uses complete-game placement rewards')
    group_episodes = group_size * 100
    if type(episode_start) is not int or episode_start < 0 or episode_start % group_episodes:
        raise ValueError('GRPO must start at a complete group boundary')
    if (type(episode_count) is not int or not 1 <= episode_count <= 10000
            or episode_count % group_episodes):
        raise ValueError('GRPO rollouts must contain complete groups and at most 10,000 episodes')
    return group_size, group_episodes


def collect_rollout(actor, opponents, config, episode_start, episode_count,
                    work_dir, check_pause=None):
    """Freeze one actor, repeat scenarios, then attach group-relative advantages.

    Environment seeds, seats and opponents match within each group. Only the
    learner's sampling seed changes. Opponents can still respond differently
    when the learner takes different actions; that is part of the game.
    """
    group_size, group_episodes = validate_groups(config, episode_start, episode_count)
    sizes, _ = rollout_settings(config, opponents)
    actor = frozen_cpu_model(actor)
    transitions = []
    hand_summaries = []
    hands_by_size = {str(size): 0 for size in sizes}
    chips_by_size = {str(size): 0 for size in sizes}
    group_summaries = []
    game_rewards = []
    for cursor in range(episode_start, episode_start + episode_count, group_episodes):
        group_number = cursor // group_episodes
        table_size = sizes[group_number % len(sizes)]
        offset = (group_number // len(sizes)) % table_size
        seed = f"grpo:train:{config.get('seed', 42)}:group:{group_number}"
        lineup = selected_opponents(opponents, table_size - 1, seed + ':lineup')
        games = []
        for repetition in range(group_size):
            if check_pause is not None:
                check_pause()
            game = play_game(
                actor, None, lineup, table_size, 100, seed, offset,
                cursor + repetition * 100, work_dir, check_pause,
                reward_config=config, learner_sample_seed=f'{seed}:sample:{repetition}',
            )
            if game['episodes'] != 100 or game.get('failures', 0):
                raise ValueError('GRPO rejected an incomplete or failed game')
            games.append(game)
        rewards = [game['objective']['reward'] for game in games]
        advantages, reward_std = group_advantages(rewards)
        group_summaries.append({'group_id': group_number, 'table_size': table_size,
                                'seed': seed, 'offset': offset, 'rewards': rewards,
                                'reward_std': reward_std, 'zero_spread': reward_std <= 1e-8})
        for repetition, game in enumerate(games):
            decision_count = len(game['transitions'])
            for transition in game['transitions']:
                # One complete game is one trajectory. Every decision carries
                # its group's relative result, with equal game weighting later.
                transition.update(group_id=group_number,
                                  trajectory_id=group_number * group_size + repetition,
                                  trajectory_decisions=decision_count,
                                  advantage=advantages[repetition],
                                  informative_group=reward_std > 1e-8)
                transition.pop('value', None)
                transitions.append(transition)
            hand_summaries.extend(game['hand_summaries'])
            hands_by_size[str(table_size)] += game['episodes']
            chips_by_size[str(table_size)] += game['chips']
        game_rewards.extend(rewards)
    groups = len(group_summaries)
    return {'transitions': transitions, 'hand_summaries': hand_summaries,
            'episodes': len(hand_summaries), 'hands_by_size': hands_by_size,
            'chips_by_size': chips_by_size, 'failures': 0, 'groups': groups,
            'group_summaries': group_summaries, 'games': len(game_rewards),
            'game_rewards': game_rewards, 'train_reward': sum(game_rewards) / len(game_rewards),
            'reward_std': sum(group['reward_std'] for group in group_summaries) / groups,
            'zero_spread_fraction': sum(group['zero_spread'] for group in group_summaries) / groups}
