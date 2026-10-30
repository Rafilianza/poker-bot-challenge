# Poker Bot Challenge Competition

Build, train and compare poker bots in a local dashboard. Explore behaviour cloning,
PPO, self-play PPO and GRPO across 4-, 5- and 6-player tables.

**Collect agents → Build datasets → Train → Evaluate**

| Dashboard | Real SELF-PPO reward history |
| --- | --- |
| ![Dashboard overview and navigation](docs/images/dashboard.png) | ![Recorded SELF-PPO reward curves at 479,000 episodes](docs/images/self-ppo-rewards.png) |

Real local screenshots. Fresh installs start empty; trained weights, collected
bots and private datasets are not bundled. Three simple example bots are included.

## Try it

Use **Python 3.12**. If `python` is unavailable or points to another version,
use `python3.12` on macOS/Linux or `py -3.12` on Windows to create the environment.

```bash
git clone https://github.com/Rafilianza/poker-bot-challenge.git
cd poker-bot-challenge
python -m venv .venv
```

Activate it:

| macOS / Linux | Windows PowerShell |
| --- | --- |
| `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |

Then launch:

```bash
python -m pip install -r requirements.txt
python run.py --open-browser
```

Open **http://127.0.0.1:8765**. No GPU or pretrained model is needed for the demo.

**First run:** upload the three Python files from `examples/` in **Model library**,
select them, then choose **Evaluate → Benchmark → 10 hands**.

## Experiment progress

BC and PPO experiments have completed. The latest recorded **SELF-PPO** run is
paused at **479,000 / 600,000 episodes**. GRPO is implemented but has no recorded
training run. See the [experiment history](docs/EXPERIMENTS.md) for results,
including earlier lookup and search work.

## Learn more

- [Setup, optional training & saved reward graphs](docs/SETUP.md)
- [How the pipeline works](docs/PIPELINE.md)
- [Model authors, sources & credits](docs/CREDITS.md)
- [Contributing](CONTRIBUTING.md) · [Validation results](docs/VALIDATION.md)

Run locally and only import bots or checkpoints you trust.
