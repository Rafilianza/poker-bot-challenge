"""Collect poker trajectories and evaluate frozen PPO policies locally.

Legacy mode rewards complete hand profit in big blinds. Placement mode rewards
the final standings of a complete 100-hand game. Features use legal observations
and, for placement policies, completed public game results.
"""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import torch
from macpoker.actions import Action
from macpoker.match import MatchConfig, MatchRunner
from macpoker.transport import SubprocessTransport, Transport

from arena.bc_features import action_options, encode_observation


class RolloutError(RuntimeError):
    """An invalid game must not contribute PPO training examples or scores."""


class OpponentTransport(SubprocessTransport):
    """Let each SDK reader close its own pipe after the subprocess exits."""

    def _read_stdout(self):
        with self.proc.stdout:
            super()._read_stdout()

    def _read_stderr(self):
        with self.proc.stderr:
            super()._read_stderr()


def seed_number(value):
    digest = hashlib.sha256(str(value).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


def frozen_cpu_model(model):
    if model is None:
        return None
    frozen = deepcopy(model).to("cpu")
    frozen.eval()
    frozen.requires_grad_(False)
    return frozen


class LearnerTransport(Transport):
    """Adapt a frozen actor and critic to the official engine's transport API."""

    def __init__(self, actor, critic, seed, episode_start, table_size, check_pause=None,
                 reward_mode='chips'):
        self.actor = actor
        self.critic = critic
        self.generator = torch.Generator(device="cpu").manual_seed(seed_number(seed))
        self.episode_start = episode_start
        self.table_size = table_size
        self.check_pause = check_pause
        self.transitions = []
        self.hand_summaries = []
        self.hand_transition_start = 0
        self.pending_action = None
        self.reward_mode = reward_mode
        self.context = None
        if next(actor.parameters()).shape[1] == 196:
            from arena.ppo_context import GameContext
            self.context = GameContext()

    def send(self, message):
        if self.context is not None:
            self.context.send(message)
        message_type = message["type"]
        if message_type == "hand_start":
            if self.check_pause is not None:
                self.check_pause()
            self.hand_transition_start = len(self.transitions)
            self.pending_action = None
        elif message_type == "action" and self.pending_action is not None:
            requested = self.pending_action
            if message["hand"] != requested["hand"] or message["seat"] != requested["seat"]:
                return
            if message["action"] != requested["action"]:
                raise RolloutError("The engine changed the learner's selected action")
            if requested["action"] == "raise" and message["amount"] != requested["amount"]:
                raise RolloutError("The engine changed the learner's selected raise amount")
            self.pending_action = None
        elif message_type == "hand_end":
            # Player zero is the learner; its seat changes each hand.
            seat = message["players"].index(0)
            chips = message["deltas"][seat]
            reward = chips / 2.0
            decisions = len(self.transitions) - self.hand_transition_start
            if decisions and self.reward_mode == 'chips':
                self.transitions[-1].update(reward=reward, done=True)
            self.hand_summaries.append({
                "episode_id": self.episode_start + message["hand"],
                "table_size": self.table_size, "seat": seat, "chips": chips,
                "reward_bb": reward, "decisions": decisions,
            })

    def act(self, observation, timeout_ms):
        if self.check_pause is not None:
            self.check_pause()
        started = time.perf_counter()
        features = self.context.encode(observation) if self.context is not None else encode_observation(observation)
        options, mask = action_options(observation)
        inputs = torch.tensor([features], dtype=torch.float32)
        legal = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits = self.actor(inputs).reshape(-1)
            if logits.numel() != len(mask) or not torch.isfinite(logits).all():
                raise RolloutError("The learner produced invalid action logits")
            logits = logits.masked_fill(~legal, -torch.inf)
            log_probabilities = torch.log_softmax(logits, dim=0)
            probabilities = torch.softmax(logits, dim=0)
            action_index = int(torch.multinomial(probabilities, 1, generator=self.generator).item())
            value = float(self.critic(inputs).reshape(-1).item()) if self.critic is not None else 0.0
        if not math.isfinite(value):
            raise RolloutError("The critic produced a non-finite value")
        self.transitions.append({
            "features": features, "mask": mask, "action": action_index,
            "old_log_prob": float(log_probabilities[action_index].item()), "value": value,
            "reward": 0.0, "done": False,
            "episode_id": self.episode_start + observation["hand"],
            "table_size": self.table_size,
        })
        chosen = options[action_index]
        self.pending_action = {**chosen, "hand": observation["hand"], "seat": observation["seat"]}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Action.from_wire(chosen), elapsed_ms


def selected_opponents(opponents, count, seed):
    if not opponents:
        raise ValueError("Choose at least one frozen opponent")
    shuffled = list(opponents)
    random.Random(str(seed)).shuffle(shuffled)
    # A small pool can occupy several seats using independent working copies.
    return [shuffled[index % len(shuffled)] for index in range(count)]


def play_game(actor, critic, opponents, table_size, hands, seed, offset,
              episode_start, work_dir, check_pause=None, reward_config=None,
              learner_sample_seed=None):
    """Run a bounded official-engine game and always close its subprocesses."""
    if (reward_config or {}).get('reward_mode') == 'game_placement' and hands != 100:
        raise ValueError('Game placement requires a complete 100-hand game')
    if check_pause is not None:
        check_pause()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # GRPO repeats an environment with independent learner action sampling.
    # Omitting this argument preserves the original PPO/evaluation stream.
    sample_seed = f"{seed}:learner:{offset}" if learner_sample_seed is None else learner_sample_seed
    learner = LearnerTransport(actor, critic, sample_seed,
                               episode_start, table_size, check_pause,
                               (reward_config or {}).get('reward_mode', 'chips'))
    transports = [learner]
    runner_path = Path(__file__).with_name("bot_runner.py")
    with tempfile.TemporaryDirectory(prefix="game-", dir=work_dir) as temporary:
        try:
            for player, opponent in enumerate(opponents, start=1):
                if opponent.get("kind") in ("current", "historical"):
                    from arena.ppo_selfplay import PolicyOpponentTransport
                    transports.append(PolicyOpponentTransport(
                        opponent["actor"], f"{seed}:policy-opponent:{player}", check_pause))
                    continue
                source = Path(opponent["path"])
                source_folder = source.parent if source.is_file() else source
                if not (source_folder / "main.py").is_file():
                    raise ValueError("Frozen opponent is missing main.py: " + opponent["name"])
                destination = Path(temporary) / f"opponent-{player}"
                shutil.copytree(source_folder, destination)
                command = [sys.executable, "-B", "-u", str(runner_path),
                           str(destination / "main.py"), f"{seed}:opponent:{player}"]
                transports.append(OpponentTransport(command))
            config = MatchConfig(seats=table_size, deals=hands, offset=offset,
                                 seed=seed, stack=200, sb=1, bb=2)
            result = MatchRunner(config, transports).run()
            if any(verdict != "OK" for verdict in result.verdicts):
                raise RolloutError("Game rejected because an agent failed: " + ", ".join(result.verdicts))
            if len(learner.hand_summaries) != hands:
                raise RolloutError("The game did not finish every requested hand")
            if sum(hand["chips"] for hand in learner.hand_summaries) != result.chips[0]:
                raise RolloutError("Learner rewards disagree with the official game result")
            for hand, summary in zip(result.hands, learner.hand_summaries):
                if hand["deltas_by_bot"][0] != summary["chips"]:
                    raise RolloutError("Learner hand reward used the wrong seat")
            game = {"transitions": learner.transitions, "hand_summaries": learner.hand_summaries,
                    "episodes": hands, "chips": result.chips[0], "failures": 0,
                    "all_chips": list(result.chips), "verdicts": list(result.verdicts),
                    "completed_hands": len(result.hands), "completed": True}
            if learner.reward_mode == 'game_placement':
                from arena.ppo_objectives import game_reward
                objective = game_reward(result.chips, hands=hands,
                                        placement_weight=reward_config.get('placement_weight', 0.9),
                                        chip_scale_bb100=reward_config.get('chip_scale_bb100', 100))
                # The entire game is one trajectory. This also handles a
                # final hand where the learner did not need to act.
                if learner.transitions:
                    learner.transitions[-1].update(reward=objective['reward'], done=True)
                game['objective'] = objective
            return game
        finally:
            for transport in transports:
                transport.close()


def rollout_settings(config, opponents, allow_self_play=False):
    sizes = config.get("table_sizes", [4, 5, 6])
    if (not isinstance(sizes, list) or not sizes
            or any(type(size) is not int or size not in (4, 5, 6) for size in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("Choose distinct PPO table sizes from 4, 5 and 6")
    if not opponents and not (allow_self_play and config.get('training_mode') == 'self_play'):
        raise ValueError("Choose at least one frozen opponent")
    mode = config.get('reward_mode', 'chips')
    if mode not in ('chips', 'game_placement'):
        raise ValueError('Unknown PPO reward mode')
    if mode == 'game_placement':
        return sorted(sizes), 100
    game_hands = config.get("game_hands", 10)
    if type(game_hands) is not int or not 1 <= game_hands <= 20:
        raise ValueError("Rollout games must contain 1–20 hands")
    return sorted(sizes), game_hands


def collect_rollout(actor, critic, opponents, config, episode_start, episode_count,
                    work_dir, check_pause=None, self_play_history=None):
    """Collect exactly episode_count complete hands with one frozen policy."""
    if type(episode_start) is not int or episode_start < 0:
        raise ValueError("Episode start must be a non-negative integer")
    if type(episode_count) is not int or not 1 <= episode_count <= 10000:
        raise ValueError("A rollout must request 1–10,000 hands")
    training_mode = config.get('training_mode', 'bots')
    if training_mode not in ('bots', 'self_play', 'mixed'):
        raise ValueError('Unknown PPO training mode')
    sizes, game_hands = rollout_settings(config, opponents, allow_self_play=True)
    placement_mode = config.get('reward_mode', 'chips') == 'game_placement'
    if placement_mode and (episode_start % 100 or episode_count % 100):
        raise ValueError('Placement rollouts must start and end on complete 100-hand game boundaries')
    actor = frozen_cpu_model(actor)
    critic = frozen_cpu_model(critic)
    transitions = []
    hand_summaries = []
    hands_by_size = {str(size): 0 for size in sizes}
    chips_by_size = {str(size): 0 for size in sizes}
    game_rewards = []
    self_play = {"mode": training_mode, "current_iteration": config.get('self_play_iteration', 0),
                 "seat_counts": {"current": 0, "historical": 0, "bots": 0},
                 "history_iterations_used": [], "games": 0}
    episode_end = episode_start + episode_count
    cursor = episode_start
    while cursor < episode_end:
        game_number = cursor // game_hands
        table_size = sizes[game_number % len(sizes)]
        # Align chunks to stable episode boundaries, including a short final game.
        hands = min(game_hands - cursor % game_hands, episode_end - cursor)
        offset = (game_number // len(sizes) + cursor % game_hands) % table_size
        seed = f"ppo:train:{config.get('seed', 42)}:episode:{cursor}"
        if training_mode == 'bots':
            lineup = selected_opponents(opponents, table_size - 1, seed + ":lineup")
        else:
            from arena.ppo_selfplay import training_opponents
            lineup = training_opponents(actor, opponents, self_play_history, table_size - 1,
                                        seed + ':lineup', training_mode,
                                        self_play['current_iteration'])
            self_play['games'] += 1
            for opponent in lineup:
                kind = opponent.get('kind', 'bots')
                self_play['seat_counts'][kind] += 1
                if kind == 'historical':
                    self_play['history_iterations_used'].append(opponent['iteration'])
        game_options = {'reward_config': config} if placement_mode else {}
        game = play_game(actor, critic, lineup, table_size, hands, seed, offset,
                         cursor, work_dir, check_pause, **game_options)
        transitions.extend(game["transitions"])
        hand_summaries.extend(game["hand_summaries"])
        hands_by_size[str(table_size)] += game["episodes"]
        chips_by_size[str(table_size)] += game["chips"]
        if placement_mode:
            game_rewards.append(game['objective']['reward'])
        cursor += hands
    result = {"transitions": transitions, "hand_summaries": hand_summaries,
            "episodes": len(hand_summaries), "hands_by_size": hands_by_size,
            "chips_by_size": chips_by_size, "failures": 0}
    if placement_mode:
        result.update(train_reward=sum(game_rewards) / len(game_rewards),
                      games=len(game_rewards), game_rewards=game_rewards)
    if training_mode != 'bots':
        self_play['history_iterations_used'] = sorted(set(self_play['history_iterations_used']))
        result['self_play'] = self_play
    return result


def evaluate_policy(actor, opponents, config, work_dir, split="validation", check_pause=None):
    """Evaluate matched decks and all seats; validation and test seeds are disjoint."""
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    if config.get('reward_mode', 'chips') == 'game_placement':
        return evaluate_placement_policy(actor, opponents, config, work_dir, split, check_pause)
    sizes, game_hands = rollout_settings(config, opponents)
    eval_hands = config.get("eval_hands", 10)
    eval_seeds = config.get("eval_seeds", 3)
    if type(eval_hands) is not int or not 1 <= eval_hands <= 1000:
        raise ValueError("Evaluation needs 1–1000 hands per seat rotation")
    if type(eval_seeds) is not int or not 1 <= eval_seeds <= 20:
        raise ValueError("Evaluation needs 1–20 seed groups")
    actor = frozen_cpu_model(actor)
    totals = {str(size): {"chips": 0, "hands": 0, "failures": 0} for size in sizes}
    seed_results = []
    episode_number = 0
    for group_number in range(eval_seeds):
        group = f"ppo:{split}:{config.get('seed', 42)}:group:{group_number}"
        group_rates = {}
        for table_size in sizes:
            row = totals[str(table_size)]
            group_chips = 0
            group_hands = 0
            lineup = selected_opponents(opponents, table_size - 1, f"{group}:size:{table_size}:lineup")
            for offset in range(table_size):
                for hand_start in range(0, eval_hands, game_hands):
                    hands = min(game_hands, eval_hands - hand_start)
                    seed = f"{group}:size:{table_size}:chunk:{hand_start}"
                    game = play_game(actor, None, lineup, table_size, hands, seed, (offset + hand_start) % table_size,
                                     episode_number, work_dir, check_pause)
                    group_chips += game["chips"]
                    group_hands += game["episodes"]
                    episode_number += hands
            row["chips"] += group_chips
            row["hands"] += group_hands
            group_rates[str(table_size)] = group_chips / 2.0 / group_hands * 100.0
        seed_results.append({"seed_group": group, "bb100": sum(group_rates.values()) / len(sizes),
                             "by_table_size": group_rates})
    for row in totals.values():
        row["bb100"] = row["chips"] / 2.0 / row["hands"] * 100.0
    rates = [row["bb100"] for row in seed_results]
    return {
        "split": split, "bb100": sum(row["bb100"] for row in totals.values()) / len(sizes),
        "aggregation": "Equal mean of the selected table-size bb/100 rates",
        "by_table_size": totals, "hands": episode_number,
        "chips": sum(row["chips"] for row in totals.values()), "failures": 0,
        "seed_groups": [row["seed_group"] for row in seed_results], "by_seed_group": seed_results,
        "seed_group_variation": {"minimum_bb100": min(rates), "maximum_bb100": max(rates),
                                 "description": "Descriptive seed-group range; not a confidence interval"},
    }


def evaluate_placement_policy(actor, opponents, config, work_dir, split, check_pause=None):
    """Score complete duplicate rounds of uninterrupted 100-hand SDK games."""
    from arena.ppo_objectives import aggregate_rounds
    sizes, _ = rollout_settings(config, opponents)
    if config.get('eval_hands', 100) != 100:
        raise ValueError('Placement evaluation requires complete 100-hand games')
    seeds = config.get('eval_seeds', 3)
    if type(seeds) is not int or not 1 <= seeds <= 20:
        raise ValueError('Evaluation needs 1–20 seed groups')
    actor = frozen_cpu_model(actor)
    games = []
    groups = []
    for group_number in range(seeds):
        group = f"ppo:{split}:{config.get('seed', 42)}:group:{group_number}"
        groups.append(group)
        for size in sizes:
            seed = f'{group}:size:{size}'
            lineup = selected_opponents(opponents, size - 1, seed + ':lineup')
            for offset in range(size):
                game = play_game(actor, None, lineup, size, 100, seed, offset,
                                 len(games) * 100, work_dir, check_pause, reward_config=config)
                games.append({'seed_group': group, 'table_size': size, 'offset': offset,
                              'seed': seed, 'chips': game['all_chips'], 'verdicts': game['verdicts'],
                              'completed_hands': game['completed_hands'], 'completed': game['completed'],
                              'opponents': [opponent['id'] for opponent in lineup]})
    result = aggregate_rounds(games, placement_weight=config.get('placement_weight', 0.9),
                              chip_scale_bb100=config.get('chip_scale_bb100', 100),
                              expected_seeds=groups, table_sizes=sizes)
    if not result['eligible_for_best'] or result['selection_score'] is None:
        raise RolloutError('Placement evaluation did not finish every clean duplicate round')
    rates = [row['bb100'] for row in result['by_seed_group']]
    result.update(split=split, seed_groups=groups, games=games,
                  reward_mode='game_placement',
                  aggregation='Equal table-size mean of exact round placement plus bounded game chips',
                  seed_group_variation={'minimum_bb100': min(rates), 'maximum_bb100': max(rates),
                                        'description': 'Descriptive seed-group range; not a confidence interval'})
    return result
