# Credits and model sources

This project is an experiment dashboard and training pipeline. Collected bots are
credited to their original authors below. Adapting their interface, recording their
decisions, or training a student from those decisions does not make the original
strategy our work.

The first release contains **references, not the collected bot source, pretrained
weights, original datasets, or private run history**. The machine-readable
[source registry](model-sources.json) records exact revisions where available.
An upstream download is not automatically a compatible dashboard upload: the
local adapters described here translate observations and legal actions.

## Collected policies and pretrained models

“Fullhouse Bot” below is Advit Arora's learned model. It is distinct from the
Fullhouse Hackathon engine's Aggressor, Mathematician, and Shark examples.

| Local name | Original author / credited project | Exact source | What was used and changed |
| --- | --- | --- | --- |
| Fullhouse Bot | Advit Arora (`advitrocks9`) | [fullhouse-bot, `e504793`](https://github.com/advitrocks9/fullhouse-bot/tree/e504793d480b1b975f25258d25939b45c6dbd5a4) | `bot/bot.py`, feature encoder, `deep_cfr_lookup.py`, `deep_cfr_model.npz`, and `preflop_equity.npz`. Local adapter translates chip units, observation/history format, legal actions, imports, and random state. The original trained weights were used without retraining. |
| Tumbleweed Gunslinger | Timo Janssen (`TimoJanssen1`) | [tumbleweed, `38b701d`](https://github.com/TimoJanssen1/tumbleweed/tree/38b701d907b2fe1caf7149c8eedfec5c29b77a77) | `bots/gunslinger/bot.py`, upstream name `tumbleweeddutch_v24`. Rule-based decisions, Monte Carlo equity, and public opponent statistics. Local changes cover chip scaling, actual streets, legal actions, self-exclusion, and per-instance state. No neural weights. |
| V16 | `lucashsu007-create` | [fullhouse-hackathon-write-up-repo, `8b98ee3`](https://github.com/lucashsu007-create/fullhouse-hackathon-write-up-repo/tree/8b98ee3efce3b224177609352258dc86db901cce) | `hero_bots/bot_v16_squeeze_C_metacap/bot.py`. Local changes translate chip units and public history, isolate state, and account for multiway split pots. No neural weights. |
| Talibus | `Taliwanmli` | [Talibus-Deep-CFR-Poker-AI, `1fcdced`](https://github.com/Taliwanmli/Talibus-Deep-CFR-Poker-AI/tree/1fcdced776e81914c3f9d37ee1606cb8d46c659d) | Release `talibus-6max-longrun-opt-v1`, `strategy_shared_best_ring.onnx`. Python adapter implements the original 510-feature encoding and 13 action slots, then maps them to legal SDK actions. Uses pretrained policy inference; the Rust search system is not included. |
| GTOBot | `luckyone18/Poker` contributors | [Poker, `c3e4e05`](https://github.com/luckyone18/Poker/tree/c3e4e055ecdcdf40183201ad368fbd7f1e0603d7) | `GTOBot` in `bots/gto_bot.py`. Heuristic mixed strategy, not a solved GTO policy. Local changes replace evaluation/equity helpers, handle split pots, and translate SDK states and actions. |
| MonteCarloBot | `luckyone18/Poker` contributors | [same pinned repository](https://github.com/luckyone18/Poker/tree/c3e4e055ecdcdf40183201ad368fbd7f1e0603d7) | `MonteCarloBot` in `bots/monte_carlo_bot.py`. Local factory uses 200 equity trials; changes include independent randomness, split pots, evaluation, and SDK adaptation. |
| SmartBot | `luckyone18/Poker` contributors | [same pinned repository](https://github.com/luckyone18/Poker/tree/c3e4e055ecdcdf40183201ad368fbd7f1e0603d7) | `SmartBot` in `bots/poker_mind_bot.py`. Local factory uses 100 equity trials, independent randomness, split-pot handling, and SDK adaptation. |

The archived license notices identify Gunslinger, V16, and Talibus as MIT.
Fullhouse's [license at the pinned revision](https://github.com/advitrocks9/fullhouse-bot/blob/e504793d480b1b975f25258d25939b45c6dbd5a4/LICENSE)
grants MIT terms to the repository author's work and explicitly excludes its
vendored engine. Its wording names specific code directories; do not assume that
this alone resolves the separate provenance of every bundled model or data file.
No license was found in the pinned GTOBot/MonteCarloBot/SmartBot source record:
**those adaptations remain reference-only unless permission is established**.

## Additional public baselines used locally

These are adaptations of decision policies, not ten pretrained neural networks.

| Local name | Original source and credit | Local adaptation |
| --- | --- | --- |
| Fullhouse Aggressor | [Fullhouse Hackathon engine](https://github.com/mbatv/fullhouse-engine/tree/a80b26957db5f5515d9273a15d1b73aadf8985db/bots/aggressor), copyright Fullhouse Hackathon, MIT | Preserves upstream decision function; translates MAC observations and clamps legal actions. |
| Fullhouse Mathematician | [same engine, mathematician](https://github.com/mbatv/fullhouse-engine/tree/a80b26957db5f5515d9273a15d1b73aadf8985db/bots/mathematician), MIT | Preserves the pot-odds caller and translates the interface. |
| Fullhouse Shark | [same engine, shark](https://github.com/mbatv/fullhouse-engine/tree/a80b26957db5f5515d9273a15d1b73aadf8985db/bots/shark), MIT | Preserves upstream preflop/position rules, including documented quirks; translates the interface. |
| PyPokerEngine Honest | [`ishikota/PyPokerEngine`, `a52a048`](https://github.com/ishikota/PyPokerEngine/blob/a52a048a15da276005eca4acae96fb6eeb4dc034/examples/players/honest_player.py), copyright `ishikota`, MIT | Calls when estimated win probability is at least `1 / table_size`. Local equity evaluator; 1,000 upstream trials capped at 128 for clock feasibility. |
| RLCard Rule V1 | [`datamllab/rlcard`, `d7d0a95`](https://github.com/datamllab/rlcard/blob/d7d0a957baf4cc7225a50522adb0164bf130a9d0/rlcard/models/limitholdem_rule_models.py), copyright DATA Lab at Texas A&M University, MIT | Original rule class with translated cards and legal actions; a fixed-limit raise maps to a minimum no-limit raise. This changes the game setting. |
| DeepBot Conservative | [`tamlhp/deepbot-poker`, `515246b`](https://github.com/tamlhp/deepbot-poker/tree/515246b5fa259e7455f8169d28f89a208694dc4c/code/bots), `cyril` in source headers and project contributors, Apache-2.0 | Ports the rank-based calling rule to the SDK. |
| DeepBot Maniac | Same pinned DeepBot source and credit, Apache-2.0 | Ports the pot/double-pot aggression rule with legal raise translation. |
| DeepBot Equity | Same pinned DeepBot source and credit, Apache-2.0 | Ports equity thresholds; replaces the native equity estimator with 100 local Monte Carlo trials and fractional split-pot credit. |
| DeepBot Candid | Same pinned DeepBot source and credit, Apache-2.0 | Ports equity-to-the-seventh-power sizing and branch order; replaces native equity estimation and translates stack/bet units. |
| DeepBot PStrat | Same pinned DeepBot source and credit, Apache-2.0 | Ports range, position, stack, and sizing rules; translates hand evaluation and actions. Documented upstream quirks are retained where possible. |

The initial locally registered bot named **“lil fruit”** carries the DeepBot Equity
adaptation notice. This mapping applies to that inspected snapshot only. Later
“lil fruit”, “Invoker”, replay-proxy, repaired, and trained variants require their
own lineage records; matching a name is not evidence of identical code or rights.
The provided collection archive is not bundled with this release.

## Datasets and learned descendants

The [PHH dataset](https://github.com/uoftcprg/phh-dataset) is credited to the
**Universal, Open, Free, and Transparent Computer Poker Research Group**. The
inspected checkout is revision `e47fbd5816372360bade4de5d712346fe1bb70f6`, with an
MIT notice for 2024–2025. Its own README attributes constituent histories to
HandHQ, the Pluribus study by Noam Brown and Tuomas Sandholm, televised WSOP hands,
and historical sources. The full dataset release is linked through
[Zenodo](https://doi.org/10.5281/zenodo.10796885); its contents differ from the
GitHub subset. Record the actual subset, revision, and license used for a run.

PHH lookup tables are local aggregates of demonstrations. “Pluribus” in a lookup
name describes a hand-history source; it does **not** mean that Pluribus's original
implementation or pretrained weights were obtained. Local BC, PPO, self-play PPO,
GRPO, and PPO-plus-search models are separate experiments. Their future model
cards should list parent checkpoint hashes, teacher bots and source revisions,
dataset licenses, training settings, and changes. A trained student is not the
original teacher model.

The frozen local checkpoint `PPO_5p_600_000_v3.zip` has SHA-256
`8de38bba8e550fbfa18a67195ecb32d7acc821363216d851645451e9d477168c`.
The four local PPO variants share the same weight hash
`212d13bea904f153859aa03889b29512fb5e71bf96d782edd465efcd9f988e75`.
Those hashes establish artifact identity, not a complete training-data lineage
or a redistribution license. Checkpoints and derived lookup tables are excluded
from this release.

## SDK and installed dependencies

The engine is **macpoker 0.1.0**, supplied by the MAC poker tournament organizers
through the [official SDK wheel](https://poker.monashcoding.com/dl/macpoker-0.1.0-py3-none-any.whl)
and [official documentation](https://docs.poker.monashcoding.com).
The installed wheel declares MIT in its package metadata. The SDK is installed
as a dependency, not copied into this repository. The official download inspected
for this release has SHA-256
`75904f0d557fff23ab25d8257cac9cd2e32742749c3e8fbd1d2e2fe5bb5e6f4b`.
The earlier development install came from wheel hash
`f816cef6e828650d4b035e3c11238484b19f341e4c3eb52d5103ab4a480772cc`.
The organizer replaced the same-version download: the inspected source changes
raise the SDK's maximum table size from nine to ten, including CLI help and the
engine assertion. Every other Python source file is unchanged. This dashboard's
4/5/6-player experiment convention still counts all players, including the bot.
Pin the download hash as well as the version to detect future replacements.

NumPy, PyTorch, and other optional runtime dependencies retain their own authors
and licenses; installing a package does not relicense it under this project's
terms. See the dependency files for what each workflow installs.

## Adding another collected bot

Add its author or verified handle, repository URL, exact revision, policy/model
path, checksum for weights, license evidence, and local modifications to
`model-sources.json`. Keep original license/copyright notices and any required
NOTICE file with redistributed source or weights. Record changed files for
adaptations. If the license or origin is unresolved, keep the item reference-only.
Do not remove notices to produce a smaller upload ZIP.

This inventory was assembled on 11 October 2026 from local pinned-source records,
archived license notices, SDK metadata, and available upstream pages. A source
link documents origin; it does not imply endorsement or reproduced performance.
