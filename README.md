# Poker Bot Challenge Competition

Build, train and compare poker bots in a local dashboard. Explore behaviour cloning,
PPO, self-play PPO and GRPO across 4-, 5- and 6-player tables.

**Collect agents → Build datasets → Train → Evaluate**

| Dashboard | Real SELF-PPO reward history |
| --- | --- |
| ![Dashboard overview and navigation](docs/images/dashboard.png) | ![Recorded SELF-PPO reward curves at 479,000 episodes](docs/images/self-ppo-rewards.png) |

Real local screenshots. Fresh installs start empty; trained weights, collected
bots and private datasets are not bundled. Three simple example bots are included.

## General flow

1. **Register existing bots.** Add agents to the model library. They can serve as
   baselines to compare against, opponents to play against, or teachers to learn
   from. You can evaluate existing bots without training a new one.

2. **Build a dataset.** Let selected teachers play and record what they could see
   and which actions they took. Alternatively, import compatible PHH hand
   histories. The result is a collection of situation-and-action examples.

3. **Train a starting bot with behaviour cloning (BC).** The model learns to
   imitate the recorded actions. This produces a playable bot and the starting
   checkpoint for the dashboard's reinforcement-learning methods.

4. **Try learning from new games.** From a completed BC checkpoint, choose
   **PPO** against registered opponents, **self-play PPO** against copies of the
   learner, or **GRPO** to compare rewards from matched attempts. Mixed self-play
   can also use older copies and registered bots. These are alternative
   experiments; choose one route at a time. You can also keep the BC bot.

5. **Export and compare.** Use validation results to select a checkpoint, export
   it as a playable bot, and compare it with unchanged baselines on fresh seeds.
   Test the **same exported bot at 4-, 5- and 6-player tables**, counting the bot
   itself. Check each table size separately to see where it performs well or
   still needs work.

The [pipeline guide](docs/PIPELINE.md) explains the data, rewards and evaluation
rules behind each stage.

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
