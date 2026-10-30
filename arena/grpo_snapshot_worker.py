"""Validate GRPO snapshots using the same tested playable-agent exporter."""
import json
from pathlib import Path
import sys

from arena import ppo_snapshot_worker


def export(folder):
    # The shared exporter checks shape, finite weights, Python inference and SDK
    # legality. GRPO only needs its own actor-only resume metadata.
    request = json.loads((folder / 'export-request.json').read_text())
    source_metadata = request.get('source_metadata') or {}
    checkpoint_bytes = (folder / 'full.pt').read_bytes()
    checkpoint, _, _, is_bc = ppo_snapshot_worker.load_actor(checkpoint_bytes, source_metadata)
    if not is_bc and (not isinstance(checkpoint, dict) or checkpoint.get('algorithm') != 'grpo'):
        raise ValueError('This saved checkpoint is not a GRPO checkpoint')
    report = ppo_snapshot_worker.export(folder)
    required_resume_fields = {'actor', 'actor_optimizer', 'progress', 'request_hash',
                              'feature_version', 'torch_rng', 'python_rng'}
    changes = {'algorithm': 'bc' if is_bc else 'grpo',
               'checkpoint_format': 'bc' if is_bc else 'grpo',
               'resumable': not is_bc and required_resume_fields.issubset(checkpoint)}
    if request['identity']['kind'] == 'latest':
        changes['selection_metric'] = 'latest_completed_iteration'
    report['metadata'].update(changes)
    report.update(changes)
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
