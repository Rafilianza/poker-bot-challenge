# Install and run

## Core dashboard: Windows, macOS, Linux

Install Python 3.12 and Git, clone the repository, and open a terminal in the
checkout. Python 3.12 is the common target for the SDK and the test matrix.

```text
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
```

Replace the two placeholders with the repository URL and directory you cloned.
On systems where `python` is not Python 3.12, use `python3.12` or Windows
`py -3.12` for the environment-creation command.

| Shell | Activate the environment |
| --- | --- |
| macOS/Linux bash or zsh | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

Then:

```text
python -m pip install -r requirements.txt
python run.py --open-browser
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of
`python` in the two commands; changing system execution policy is unnecessary.
On macOS/Linux, `.venv/bin/python` works without activation too.

The app opens at http://127.0.0.1:8765. Stop it with Ctrl+C in its terminal.
If the port is occupied, use `python run.py --port 8767 --open-browser`.
Use `--data-dir local-data/my-study` for a separate workspace. Relative paths
passed to this launcher are resolved from the checkout, even if launched from
another directory. Restarting holds unfinished jobs; it does not silently resume
training. Use the relevant job's Resume action where supported.

The core installation needs no PyTorch, eval7, ONNX Runtime, pretrained weights,
or Node.js. Third-party bots can require additional libraries; consult their
authors' instructions before importing them.

## Optional training

BC, PPO, self-play PPO, and GRPO require PyTorch. On macOS:

```text
python -m pip install -r requirements-training.txt
```

For a smaller CPU-only installation on Windows/Linux, install the CPU build
first, then the training requirements:

```text
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
```

For NVIDIA acceleration, use the compatible installation command from
[PyTorch's official selector](https://pytorch.org/get-started/locally/), keeping
the project's required torch version. Select **NVIDIA GPU (CUDA)** in training.
For Apple Silicon, select **Apple GPU (MPS)** when available. **CPU** always
avoids a GPU requirement. **Automatic** chooses an available supported device.
An explicitly requested unavailable GPU produces an error rather than silently
changing the experiment's device.

Performance and random trajectories can differ by hardware. Record dependency
versions, device, seeds, source models, and table sizes when comparing results.

## Optional PHH import

```text
python -m pip install -r requirements-phh.txt
```

The replay converter pins PokerKit 0.7.6 because replay semantics affect labels.
Download a dataset yourself from the sources in [CREDITS.md](CREDITS.md), retain
its notices, and select its path in **Data → Import hand histories**. The default
location is `external/phh-dataset/`; a custom path also works. Datasets and the
`external/` directory are ignored by Git. No private dependency folder is needed.

## View saved PPO reward graphs

If you have an earlier Poker Arena data folder containing `arena.sqlite`, you
can view its recorded PPO and self-play curves without resuming its jobs:

```text
python run.py --port 8777 --data-dir local-data/preview --live-ppo-data "/path/to/original/arena/data"
```

Replace the example source path with your own data folder. It must be separate
from the preview's data folder. Open **Training → PPO / self-play PPO → View
saved PPO / self-play reward graphs**, or visit
`http://127.0.0.1:8777/#ppo-live-section` directly. The source is read-only;
graphs use the recorded rewards, validation results and original run statuses.
Viewing them does not copy weights. Checkpoint downloads are optional and only
run when requested. Legacy chip rewards and placement rewards retain their
different units. Historical curves require recorded job history; a model ZIP
alone does not contain that history. Private histories remain excluded from Git.

## First evaluation

Upload `examples/calling_bot.py`, `examples/cautious_bot.py`, and
`examples/random_bot.py` in **Model library**. Select them and open **Evaluate**.
Use Benchmark, 10 hands, one seed `demo-1`. Each agent plays the SDK's house
opponents with rotated seats. View progress in **Experiments**. The included
three agents are enough for a benchmark; a study covering six-player tables
needs at least six registered agents.

Runs accept up to 80 agents and 20 distinct seeds. Table sizes count the bot
itself: a six-player table contains our bot and five opponents. For comparative
experiments, keep seeds, hand count, opponents, and scoring mode consistent.

## Dependency and platform notes

The SDK comes from the competition's official wheel URL. Its hash is pinned:
the publisher previously replaced a same-version wheel, so a new unexpected
hash should be investigated before updating the requirement. Source comparison
and both known hashes are recorded in [model-sources.json](model-sources.json).
Installation needs access to PyPI and that SDK host; gameplay is local.

The release targets Python 3.12 on Windows, macOS, and Linux with compatible
binary wheels. It cannot promise support for every OS version, processor, or
GPU. CI defines checks for all three OS families; local verification is on
Apple Silicon macOS. Windows/Linux results remain unverified until CI runs.

If installation fails, confirm Python 3.12, a current pip, internet access, and
wheel support for your platform. A firewall or unavailable SDK download is a
dependency problem, not a reason to reuse somebody else's virtual environment.
