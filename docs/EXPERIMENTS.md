# Experiment record

This is a **sanitized historical snapshot as of October 11, 2026**, using Australia/Melbourne dates. It was checked against the original dashboard's source, saved job records and standalone validation reports. It is not the live job history of a new installation. Raw databases, teacher datasets, uploaded third-party bots and neural checkpoints are not bundled in this source release.

The aim is one adaptable policy that can be evaluated at **4, 5 and 6 total players**, including the bot. The project has tried imitation, reinforcement learning, lookup and limited search. No method is declared the overall winner by this record.

## Current research context

The most recently created training experiment in the audited main dashboard is **SELF-PPO**, created October 4. Its saved state is **paused at 479,000 of 600,000 hand episodes**. It starts from `BC_2`, uses pure self-play at 4/5/6-player tables, and trains with 90% game placement plus 10% bounded chip profit. Its validation-selected checkpoint is iteration 355; saved training progress is iteration 479. This unfinished run has no claimed completed final held-out test.

There were no running or queued jobs in that database when inspected. This is a statement about saved records, not a claim about every separate development script or process. A fresh clone will not resume SELF-PPO because its private training state is not included.

The audited main database contains 108 registered agent versions, 21 completed arena runs, two completed BC jobs, six completed PPO jobs, twelve paused PPO jobs, and no GRPO jobs. Registered versions can duplicate a name; names alone are not content identity. The history below focuses on the experiments with completed training or saved nonzero progress. Paused drafts with zero episodes are not counted as trained policies.

## What has been tried

| Family | Question | Actual status | Evidence and limit |
| --- | --- | --- | --- |
| Existing models | Which existing strategies are useful teachers and opponents? | Collection and arena comparisons completed | Registered source snapshots and recorded decisions exist. Attribution belongs to upstream authors; registration is not proof of permission to redistribute. |
| Behaviour cloning | Can one compact policy imitate a diverse teacher pool? | Two training runs completed | BC_1 and BC_2 used the same mixed-table dataset. Their metrics describe imitation. |
| PPO against bots | Can new poker experience improve a BC policy? | Several completed and paused runs | Both five-player-only and mixed 4/5/6 training were tried. Legacy chip and newer placement rewards are distinct experiments. |
| Self-play PPO | Can copies of one policy improve through play against each other? | SELF-PPO paused with saved progress | Pure self-play has a main experiment. Mixed self-play is implemented, but no main run uses that mode in this snapshot. |
| GRPO | Can matched game groups provide a useful update without a critic? | Implemented with focused tests; no main training job | Workflow exists, but no trained GRPO contender or measured improvement is claimed. |
| PHH lookup | Can offline demonstrations become a compact, fast policy table? | Standalone tables built and checked | V1 supports variable stack depths. A stricter full scan retained only eligible six-player Pluribus data. |
| Search | Can limited simulation improve decisions within the play clock? | Three standalone prototypes built and checked | Rollouts, single-observer ISMCTS and OOS have compatibility evidence, not established winning strength. |
| Frozen PPO hybrids | Does lookup, rollouts, or both improve one unchanged actor? | Four standalone variants built and checked | Neural weights are shared. Compatibility tests exercised all components; a robust strength ranking remains open. |

## Demonstration collection and BC

`FirstDatasetV2` completed on October 3 with **310,500 table-hand episodes** and **2,952,868 decision records**. Its requested target was 300,000 episodes; collection stopped at a complete balanced boundary. The dataset mixes 4-, 5- and 6-player tables.

The two BC runs used 2,440,065 training decisions, 256,710 validation decisions and 256,093 test decisions from this dataset. Splits are grouped by seed, rather than randomly separating related decisions.

| Student | Completed training | Selected checkpoint | Validation action agreement | Held-out action agreement |
| --- | --- | --- | ---: | ---: |
| BC_1 | 10 epochs | Epoch 10 | 71.26% | 70.83% |
| BC_2 | Finished early after 76 of 100 requested epochs | Epoch 75 | 72.34% | 71.95% |

BC_2 imitates the held-out teacher actions more often in this dataset. **These percentages are not poker win rates.** They do not show that BC_2 beats its teachers, or that predicting a common action is strategically correct. The seven-action abstraction also approximates the teachers' original raise amounts.

## PPO history

Every run below started from `BC_2`. The name identifies the original experiment; actual saved settings determine its budget and table sizes. Dates are local completion dates for completed runs and local creation dates for paused runs.

| Model / experiment | Date | Training tables | Saved hands / target | Status | Selected iteration |
| --- | --- | --- | ---: | --- | ---: |
| PPO-V1 | Oct 3 | 4/5/6 | 10,000 / 10,000 | Completed | 0 |
| PPO_5p_100_000_v1 | Oct 3 | 5 | 100,000 / 100,000 | Completed | 100 |
| PPO_465p_100_000_v1 | Oct 3 | 4/5/6 | 100,000 / 100,000 | Completed | 55 |
| PPO_5p_600_000_v3 | Oct 3 | 5 | 600,000 / 600,000 | Completed | 265 |
| PPO_456p_600_000_v3 | Oct 4 | 4/5/6 | 600,000 / 600,000 | Completed | 250 |
| PPO_5p_1000_000_v3 | Oct 3 | 5 | 142,000 / 1,000,000 | Paused | 55 |
| PPO_456p_1000_000_v3 | Oct 3 | 4/5/6 | 272,000 / 1,000,000 | Paused | 255 |
| NEW REWARD BABY | Oct 4 | 4/5/6 | 500,000 / 600,000 | Paused | 270 |
| Coba dikit2 | Oct 4 | 4/5/6 | 50,000 / 50,000 | Completed | 15 |
| SELF-PPO | Oct 4 | 4/5/6 | 479,000 / 600,000 | Paused | 355 |

The first seven rows use the legacy chip reward. `NEW REWARD BABY`, `Coba dikit2` and `SELF-PPO` use the newer 90/10 placement/chip objective. The first two of those train against registered bots; SELF-PPO trains against policy copies. Scores from different objectives, table sizes or opponent pools must not be directly ranked as if they were the same measurement.

`PPO-V1` is a useful negative result: after 10,000 hands, validation still selected the initial BC policy at iteration zero. More training iterations do not guarantee that the last checkpoint is the best one. Likewise, “600,000” in an exported model name describes the completed job budget, not necessarily the training step of the exported actor.

The saved job records contain local validation and test metrics. This release does not publish a league table from incompatible profiles or a claim of statistically established improvement. A repeatable public comparison needs the actual artifacts, opponent versions, grouped results and evaluation settings released together.

## Lookup experiments

Two separate offline designs were implemented. Their generated tables and builder packages remain outside this dashboard release.

| Design | Data actually used | Decision method | Evidence |
| --- | --- | --- | --- |
| PHH_Lookup_v1 | 24,666 usable hands from Pluribus and sampled HandHQ archives; 143,915 training decisions | One frequency table with fine/coarse/general keys retaining exact player counts | 9,274 rows; direct 4/5/6 examples; saved 360-hand compatibility check |
| Six strict lookup variants | 10,000 eligible six-player Pluribus hands; 91,356 decisions before split | Pluribus-only, balanced all-sources, or weighted tables; each with full-action and fold/call-only variants | Saved 1,800-hand SDK check for the six exported agents |

V1 inspected every local source file but only the first 16 hands of each archive, alongside single-hand files. It was a bounded sample, not a full-corpus scan. Its training decisions by total players were 10,569 at four, 30,860 at five and 102,486 at six. Recoverable private cards in HandHQ can favor showdown hands.

The stricter build scanned all local records but required exact 100 BB starting stacks and complete replay/outcome agreement. Only the Pluribus collection passed its filters. Its “AllSources” label means all **eligible** sources, including human opponents in those Pluribus hands; it does not mean every scanned dataset contributed. At four and five players, its broad buckets transfer from six-player evidence.

Lookup stores observed behavior. Neither action frequency nor held-out imitation accuracy is an estimate of the action's expected profit. V1's low house-opponent coverage also showed that historical data coverage need not match gameplay coverage.

## Search and frozen PPO variants

`Search_Rollouts`, `Search_ISMCTS` and `Search_OOS` are standalone experiments with one bot file set for all supported table sizes. Each saved report covers 180 SDK hands across 4/5/6 players, opponent styles and seat rotations. Rollouts compare terminal chip results against a heuristic continuation policy. ISMCTS shares tree statistics between situations with the same information available to the bot. OOS uses sampled trajectories to update an online strategy. None is presented as a reproduction of a published championship poker system.

The four frozen variants use `PPO_5p_600_000_v3`:

| Variant | What changes at play time |
| --- | --- |
| PPO alone | Nothing: the ZIP is an exact copy of the original submission |
| PPO + lookup | Supported V1 action counts contribute at most 10% of action probabilities |
| PPO + rollouts | Simulate legal alternatives and override a sampled PPO action only when estimated gain clears a noise guard |
| PPO + lookup + rollouts | Use the blended policy for both the real baseline action and our later simulated decisions |

All weights stay frozen. The hybrids accelerate inference with NumPy, so floating-point rounding can differ from the original pure-Python forward pass. Search samples plausible hidden cards rather than observing real opponents' cards, uses approximate opponents, and estimates hand chip profit. Its default cooperative limits are 35 ms per decision and 80 ms per hand; operating-system delays can exceed them.

The saved four-variant report covers **33 focused tests and 720 SDK hands**, with legal actions, all bot verdicts OK, and zero replay errors. Lookup and search were exercised at every table size. These are historical compatibility results. They do not establish that any hybrid beats PPO alone.

## Next comparison to make

1. Freeze the candidate files and record their hashes, starting model, reward and dataset/opponent provenance.
2. Keep BC and PPO baselines unchanged. Evaluate the same candidate artifact across 4/5/6 players with mixed, passive and aggressive lineups.
3. Match seeds and seat rotations; reserve fresh opponent/seed groups for the final test. Use full 100-hand games when comparing placement objectives.
4. Report chip profit, game and round placement, completed coverage, failures and decision time separately for each table size. A positive overall average must not hide persistent losses in one configuration.
5. Quantify uncertainty using complete matched seed/deal groups. Choose the evaluation budget before inspecting the result; do not count correlated individual decisions as independent evidence.

The [pipeline guide](PIPELINE.md) explains how these workflows connect. [experiments.json](experiments.json) supplies the dashboard's historical overview. Update that file deliberately when publishing a new research snapshot; local runtime job state remains in the installation's own data directory.
