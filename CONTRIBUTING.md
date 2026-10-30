# Working on Poker Bot Challenge Competition

Start with Python 3.12 and the core setup in [README.md](README.md). The frontend
is ordinary HTML, CSS, and JavaScript; it has no build pipeline.

## Keep changes understandable

Prefer small functions, descriptive names, and straightforward Python. Keep
observations, training labels, policy inference, and evaluation separate. Never
add spectator hole cards, future board cards, or final outcomes to model inputs.

Every experiment should record its question, starting model, data/opponents,
seeds, table sizes, settings, results, and limitations. Keep local jobs in the
ignored data directory. Update `docs/experiments.json` only when deliberately
publishing a sanitized research snapshot, not after every run. It is historical
portfolio content and is never a training checkpoint.

When documenting an external model, include author, source URL, pinned revision,
exact model/policy used, license evidence, and adaptation notes. Do not remove
copyright, license, attribution, or modification notices from exports.

## Checks

```text
python scripts/check.py
```

This runs core tests and a real example-agent benchmark in a temporary data
directory. It does not touch the app's saved runs.

For learning and PHH checks, install both optional requirements and run:

```text
python scripts/check.py --full
```

Training tests use tiny fixtures and CPU. They do not reproduce the historical
large training experiments or establish playing strength.

The optional browser test starts its own isolated server and exercises
navigation, upload, a short benchmark, and mobile layout:

```text
npm ci
npx playwright install chromium
npm test
```

Node.js is needed only for this development check. The GitHub Actions workflow
runs core checks on Windows, macOS, and Linux, plus the full suite and browser
check on Linux. A configured workflow is not evidence it has passed; inspect
the actual GitHub run after publication.

Pause or finish local jobs before restarting a development server. Never commit
`arena/data`, model weights, raw hand histories, uploaded private code, or
virtual environments. No third-party credentials are required to run this app.
