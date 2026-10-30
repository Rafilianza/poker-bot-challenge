"""Frozen policy opponents for PPO training, with private state for each player."""

import random
import time

import torch
from macpoker.actions import Action
from macpoker.transport import Transport

from arena.bc_features import action_options, encode_observation
from arena.ppo_context import GameContext, PLACEMENT_FEATURE_COUNT


class PolicyOpponentTransport(Transport):
    """Sample legal actions without a critic, training records or shared memory."""

    def __init__(self, actor, seed, check_pause=None):
        # Import here because the rollout module creates these transports.
        from arena.ppo_rollout import seed_number

        self.actor = actor
        self.generator = torch.Generator(device="cpu").manual_seed(seed_number(seed))
        self.context = GameContext()
        self.uses_context = next(actor.parameters()).shape[1] == PLACEMENT_FEATURE_COUNT
        self.check_pause = check_pause
        self.pending_action = None

    def send(self, message):
        from arena.ppo_rollout import RolloutError

        self.context.send(message)
        if message["type"] == "hand_start":
            self.pending_action = None
        elif message["type"] == "action" and self.pending_action is not None:
            requested = self.pending_action
            if message["hand"] != requested["hand"] or message["seat"] != requested["seat"]:
                return
            if message["action"] != requested["action"]:
                raise RolloutError("The engine changed a self-play opponent's action")
            if requested["action"] == "raise" and message["amount"] != requested["amount"]:
                raise RolloutError("The engine changed a self-play opponent's raise amount")
            self.pending_action = None

    def act(self, observation, timeout_ms):
        from arena.ppo_rollout import RolloutError

        if self.check_pause is not None:
            self.check_pause()
        started = time.perf_counter()
        if observation["players"][observation["seat"]] != self.context.player_id:
            raise RolloutError("A self-play opponent received another player's observation")
        if self.uses_context:
            features = self.context.encode(observation)
        else:
            features = encode_observation(observation)
        options, mask = action_options(observation)
        inputs = torch.tensor([features], dtype=torch.float32)
        legal = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits = self.actor(inputs).reshape(-1)
            if logits.numel() != len(mask) or not torch.isfinite(logits).all():
                raise RolloutError("A self-play opponent produced invalid action logits")
            logits = logits.masked_fill(~legal, -torch.inf)
            probabilities = torch.softmax(logits, dim=0)
            action_index = int(torch.multinomial(probabilities, 1, generator=self.generator).item())
        chosen = options[action_index]
        self.pending_action = {**chosen, "hand": observation["hand"], "seat": observation["seat"]}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Action.from_wire(chosen), elapsed_ms


def training_opponents(actor, opponents, history, count, seed, mode, iteration):
    """Choose independently per seat: 50% current, 25% history, 25% fixed bot."""
    if mode not in ("self_play", "mixed"):
        raise ValueError("Policy opponents require self-play or mixed training")
    if mode == "mixed" and not opponents:
        raise ValueError("Mixed self-play requires registered opponents")
    generator = random.Random(str(seed))
    history = history or []
    lineup = []
    for player in range(count):
        choice = generator.random() if mode == "mixed" else 0.0
        if choice >= 0.75:
            lineup.append(generator.choice(opponents))
        elif choice >= 0.5 and history:
            saved_policy = generator.choice(history)
            lineup.append({"kind": "historical", "actor": saved_policy["actor"],
                           "iteration": saved_policy["iteration"]})
        else:
            # Until an older policy exists, its share uses the current copy.
            lineup.append({"kind": "current", "actor": actor, "iteration": iteration})
    return lineup
