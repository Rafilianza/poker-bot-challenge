# Release validation

Checked on **11 October 2026**, on Apple Silicon macOS with a newly created
Python 3.12 virtual environment in the release checkout. The original dashboard,
database, environments, trained models, and datasets were not used as mutable
test fixtures or changed by this cleanup.

## Executed locally

| Check | Result |
| --- | --- |
| Fresh core installation | Passed with the official hash-pinned SDK and NumPy; no original virtual environment reused. |
| Core test suite | 65 tests passed before installing PyTorch, including real example-agent games in a temporary workspace. |
| Full learning and PHH suite | 287 tests passed after installing optional requirements. |
| Dependency consistency | `pip check` passed. |
| Browser workflow | Passed using Playwright 1.64.0 and installed Chrome. |
| Example evaluation in the browser | 3 agents, 15 games, 150 table hands; all player verdicts `OK`. |
| Navigation | Six views, three training tabs, preserved deep links, and ZIP download passed. |
| Published research catalogue | Eight method summaries, thirteen experiment records, and seventeen model-source entries rendered. |
| Responsive layout | Desktop 1440 px and mobile 390 px checked; no page-width overflow or JavaScript errors. Screenshots visually reviewed. |
| Source hygiene | No trained weights, private database, raw dataset, machine-specific user paths, or virtual environments in the release file set. |

The full suite covers feature information boundaries, grouped dataset splits,
training recovery, PPO/self-play/GRPO updates, PHH replay, snapshots, recording,
and evaluation. Export checks now require preserving source text and license
notices. Explicit CUDA selection is checked with mocked device availability;
the run does not claim a real NVIDIA GPU test.

Core dependency versions: `macpoker==0.1.0`, `numpy==2.5.3`. Optional versions:
`torch==2.14.1`, `pokerkit==0.7.6`. SDK wheel SHA-256:
`75904f0d557fff23ab25d8257cac9cd2e32742749c3e8fbd1d2e2fe5bb5e6f4b`.

## Read-only training history check — 29 October 2026

The preview's read-only PPO source now includes recorded reward curves and
validation/test summaries. Ten snapshot tests passed, including source-file
preservation and exact history/metric transfer. Against the original local
database, all 18 source jobs retained their saved status, settings and history;
10 had recorded curves and 8 had no training measurements. SELF-PPO remained
paused at 479,000 of 600,000 episodes with 479 recorded iterations.

The browser rendered 18 source cards and 10 charts, with no source pause/resume
controls, no console errors, and no horizontal overflow at its 392 px viewport.
All 12 fingerprinted source files, including the database, remained byte-identical.
No training was started and no weights were copied. These historical results
are available through an explicitly configured local source; the private
database is not bundled with the GitHub release.

## Publication review — 31 October 2026

The 65 core tests passed again in the existing local Python environment.
The README now uses the current competition name and verified dashboard/reward
screenshots, and documents the optional read-only history viewer. Local document
links resolve. The staged release excludes weights, private run databases, raw
datasets and environments. This review was not a fresh dependency installation
or a Windows/Linux run.

## What remains unverified

The GitHub Actions workflow defines native Windows, macOS, and Linux core
checks, plus Linux training/PHH/browser checks. It has not run on GitHub yet.
Native Windows/Linux execution and real CUDA training are therefore unverified.
These compatibility tests do not reproduce the author's long training runs or
prove any agent's playing strength.

The original historical experiments remain documented separately in
[EXPERIMENTS.md](EXPERIMENTS.md). This validation concerns the cleaned release.

## Repeat

```text
python scripts/check.py
python scripts/check.py --full
npm ci
npx playwright install chromium
npm test
```

See [SETUP.md](SETUP.md) for dependencies and [CONTRIBUTING.md](../CONTRIBUTING.md)
for how the checks isolate their data and subprocesses.
