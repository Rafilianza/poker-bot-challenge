"""Small PPO networks with an exact, optional BC-compatible context extension."""
from arena.bc_features import ACTION_NAMES, FEATURE_COUNT
from arena.ppo_context import PLACEMENT_FEATURE_COUNT


def build_actor(source_state=None, context=False):
    """Load BC unchanged, or append zero-weight public context to its input."""
    import torch

    input_count = PLACEMENT_FEATURE_COUNT if context else FEATURE_COUNT
    actor = torch.nn.Sequential(
        torch.nn.Linear(input_count, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, len(ACTION_NAMES)),
    )
    if source_state is not None:
        state = dict(source_state)
        first_weights = state.get('0.weight')
        if context and torch.is_tensor(first_weights) and first_weights.shape == (128, FEATURE_COUNT):
            # Keep the BC policy identical initially. New context columns can
            # learn during PPO without rewriting the source BC checkpoint.
            extended_weights = first_weights.new_zeros((128, input_count))
            extended_weights[:, :FEATURE_COUNT] = first_weights
            state['0.weight'] = extended_weights
        actor.load_state_dict(state, strict=True)
    return actor


def build_critic(context=False):
    """The critic starts fresh and observes the same inputs as its actor."""
    import torch

    input_count = PLACEMENT_FEATURE_COUNT if context else FEATURE_COUNT
    return torch.nn.Sequential(
        torch.nn.Linear(input_count, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
