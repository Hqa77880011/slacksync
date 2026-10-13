# SlackSync

Implementation of **SlackSync: Opportunity-Cost-Aware Work Insertion During Multi-Agent Dependency Waiting**.

SlackSync advances a ready job assigned to a blocked agent when it can return to its suspended task at positive expected value. The score subtracts travel, switching, congestion and expected synchronization lateness from saved required work. A deterministic checker admits only dependency-compatible tasks with a restorable position, orientation and held-object state.

## Installation

Use Python 3.11 with a CPU or CUDA PyTorch installation. In an activated virtual environment:

```bash
python -m pip install -e .
python -m pip install "overcooked-ai @ git+https://github.com/HumanCompatibleAI/overcooked_ai@739950a079cdaed5a44fcc662efc40244c205d06"
```

The native adapter uses the pinned [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) transition API. Qwen2.5-7B-Instruct generates plans locally through Transformers; no paid API key is needed. Run commands from this repository.

## Save plans and collect waiting trajectories

```bash
slacksync make-plans --split train --count 1000 --random-start --output data/train_plans.json
slacksync make-plans --split calibration --count 400 --random-start --output data/calibration_plans.json
slacksync make-plans --split validation --count 300 --random-start --output data/validation_plans.json
slacksync collect --plans data/train_plans.json --output data/train_waits.json
slacksync collect --plans data/calibration_plans.json --output data/calibration_waits.json
slacksync collect --plans data/validation_plans.json --output data/validation_waits.json
```

Training uses cramped_room and asymmetric_advantages; calibration uses coordination_ring; validation uses large_room. `--count` is per layout, so 1,000 training initial conditions produce 2,000 trajectories. Test layouts are forced_coordination and counter_circuit. `--layouts` selects a subset within the declared split. `--seed`, `--random-start`, `--model` and `--revision` determine initial states and the frozen planner.

Each saved episode contains a native initial state, causal vertices, preferred agent chains, source layout, seed, controller/cohort metadata and planner token records. Required skills include ingredient/dish fetching, potting, starting cooking, waiting for a pot, collecting and serving soup, independent onion staging and counter handoffs. Target terrain and graph references are checked before execution. Preferred chains do not create artificial causal edges; causal predecessors must reflect actual dependencies. The planner must supply at least four required serving jobs and use the native recipe list.

`collect` executes ordinary local waiting on these saved plans and derives remaining release-time targets from actual native transitions. No future state or controller ID enters the feature vector. The companion `.episodes.json` contains primitive actions and release events, making the labels inspectable. Dependencies not released within 400 ticks receive the capped value 400.

## Train and calibrate ETA

```bash
slacksync train --train-records data/train_waits.json --validation-records data/validation_waits.json --output checkpoints/eta
slacksync calibrate --records data/calibration_waits.json --checkpoint checkpoints/eta/eta.pt --type-means checkpoints/eta/type_means.json --output checkpoints/calibration.json
```

The two-layer width-64 MLP predicts nineteen ordered quantiles at 0.05 increments. Positive cumulative increments normalized to the 400-tick horizon enforce ordering. Thirty Adam epochs, learning rate 0.001 and batch 128 are specified in `configs/default.json`. Validation pinball loss selects `eta.pt`; `history.json` records fitting and `type_means.json` supplies the fixed-ETA baseline.

Calibration samples one eligible record per independent episode. The nonnegative conformal residual is the excess outside the 5%–95% interval. The finite-sample order statistic yields the saved offset at nominal 90% coverage. Lower quantiles shift down, upper ones up, and the median remains unchanged. The runtime inverts the piecewise quantile function and integrates its CDF exactly, including clipped endpoint atoms, to compute expected lateness.

## Same-plan execution and metrics

```bash
slacksync make-plans --split test --count 150 --random-start --output data/test_plans.json
slacksync run --plans data/test_plans.json --checkpoint checkpoints/eta/eta.pt --calibration checkpoints/calibration.json --method slacksync --output runs/slacksync
slacksync run --plans data/test_plans.json --checkpoint checkpoints/eta/eta.pt --calibration checkpoints/calibration.json --method wait --output runs/wait
slacksync evaluate --inputs runs/slacksync/traces.json runs/wait/traces.json --checkpoint checkpoints/eta/eta.pt --output runs/comparison.json
slacksync plot --input runs/comparison.json --output runs/figures
```

`run` uses the saved graph and makes no planner calls during replay. Every executor gets the same initial state, plan, primitive controller and checker. An inserted task is marked complete once, skipped when its chain later reaches it, and followed by return to the saved resume state. The adapter rechecks travel against current occupancy. Insertions finish at declared macro boundaries before returning; their actual duration may exceed the estimate.

The scheduler considers up to eight own-agent ready tasks by graph rank. A candidate must not be downstream of the active dependency, must have compatible read/write resources, and must preserve the held-object resume contract. Currently independent `stage_onion` and separate `start_cooking` jobs satisfy that contract; object-changing fetch, transfer and serve jobs remain required work but are not inserted. Duration includes outward travel, execution, return and dispatch. Defaults use lateness weight 2, congestion weight 1 and modeled release-risk threshold 0.10.

`traces.json` includes all primitive ticks, waiting records, candidate scores/rejections, insertion attempts, completion/return/release ticks and CPU-stage timings. Every agent-tick is useful, idle or additional switching/return travel. `evaluate` reports capped completion ticks, success by the 400-tick horizon, delivered soup rate, utilization/idle/switch shares, attempted/completed insertions, synchronization-miss rate, median ETA MAE, interval coverage/width, CPU stages and wall time. A failed episode contributes 400 completion ticks. A miss is divided by insertion attempts, not episodes.

## Baselines and controlled conditions

Baseline methods are `wait`, `greedy`, `shortest`, `critical_path`, `fixed_eta` and `uncalibrated`. The future-time `oracle` accepts `--oracle-traces runs/wait/traces.json` and uses sealed baseline release events as a privileged timing reference. Ablations are `no_return`, `no_lateness`, `no_congestion` and `no_risk`; all retain the same hard checker and actual return behavior.

Controlled episodes remain separate from native results. Generate plans with `--controlled --action-duration 2` or `4`; the wrapper inserts primitive holding delays and blocks a recorded floor cell during ticks 160–179. The trace cohort is explicitly `controlled`. Use `make-cohort` below to change wrapper conditions while retaining the saved graph and initial state.

`make-plans --controller` chooses `lex`, `random_tie`, `clockwise`, `counterclockwise`, `reactive_yield`, `persistent_reroute` or `burst_stay`. Short aliases `cautious`, `stochastic` and `lagged` are also available. Burst staying follows a two-state process; rerouting reverses path tie order after repeated collisions. These policies retain native transitions and are distinct from controlled action-duration changes. Keep controller families and seeds isolated when constructing duration-training and test packs.

Synthetic dependency graphs separately vary safe-ready-work density and timing shift. They reuse recorded wait features and release labels, then simulate only the declared required work and return costs:

```bash
slacksync make-diagnostics --wait-records runs/wait/wait_records.json --output data/diagnostics.json
slacksync run-diagnostics --input data/diagnostics.json --checkpoint checkpoints/eta/eta.pt --calibration checkpoints/calibration.json --output runs/diagnostics
slacksync plot --input runs/diagnostics/results.json --output runs/diagnostic_figures
```

`--wait-records` accepts records emitted by `collect` or the `wait_records.json` written by `run`. Density ranges from 0 to 1, timing factors are 0.5/1/2, and `--count` controls scenarios per condition. Diagnostic results are labeled `synthetic` and stored separately from kitchen outcomes.

For end-to-end correction, add `run --replan-model Qwen/Qwen2.5-7B-Instruct`. The common validity monitor detects unreachable assigned stations and preferred-chain wait cycles before requesting a replacement graph from the current native state. It enforces ten total planner calls and 20,000 tokens including the saved initial plan. Ordinary valid waiting makes no extra model call. Corrections and runtime planner ledgers are stored separately from replay-only results.

The implementation uses a native macro controller, saved local-model plans and finite Overcooked effects. Four deliveries, layout splits, calibration records and runtime cost parameters remain visible in configuration and traces. Report planner-generation costs separately from replay execution time because the plan is saved before runtime scheduling.

## Controlled cohorts, timing shift and correction cost

```bash
slacksync make-cohort --plans data/test_plans.json --cohort controlled --output data/controlled_plans.json
slacksync make-cohort --plans data/test_plans.json --cohort extreme_shift --output data/shift_plans.json
slacksync make-cohort --plans data/test_plans.json --cohort congestion --output data/congestion_plans.json
slacksync make-cohort --plans data/test_plans.json --cohort calibration_controller --output data/calibration_controller_plans.json
slacksync run --plans data/shift_plans.json --checkpoint checkpoints/eta/eta.pt --calibration checkpoints/calibration.json --method slacksync --output runs/shift
slacksync run --plans data/shift_plans.json --checkpoint checkpoints/eta/eta.pt --calibration checkpoints/calibration.json --method wait --output runs/shift_wait
slacksync evaluate --inputs runs/shift/traces.json runs/shift_wait/traces.json --checkpoint checkpoints/eta/eta.pt --output runs/shift_comparison.json
```

Controlled durations cycle through 1/2/4 ticks; extreme shift uses 1/4/8 ticks and a burst-pause exit probability of 0.10. Native, added congestion, calibration-controller and extreme-shift cohorts keep separate names in summaries. Random event draws are keyed by episode seed and environment tick, so policy-dependent control flow does not shift the event stream.

The layout split remains fixed. Use lexicographic/random-tie controllers for training, clockwise for calibration, counterclockwise for validation, and reactive-yield/persistent-reroute/burst-stay controllers for test packs. Different `--seed` ranges keep their trajectories separate. Combine saved plan JSON arrays before `collect` or `run` when using several controller packs.

For the end-to-end efficiency comparison, add `--replan-model Qwen/Qwen2.5-7B-Instruct` and compare `slacksync` against `every_wait_replan`. The latter requests a new plan at each newly encountered blocked head when no macro is in progress. Initial and runtime planner calls share the ten-call/20,000-token cap. Summaries include runtime calls, total planner calls/tokens, model service seconds and local CPU stage times.

| Analysis | Metrics |
| --- | --- |
| Native/controlled main comparison and component removals | Capped ticks, success, soup rate, useful/idle/switch fractions |
| Insertion | Attempted/completed/missed insertions and miss fraction |
| ETA and shift | Median MAE, covered/eligible record counts, interval coverage and width |
| Opportunity density | Synthetic completion, success, miss rate and ETA metrics by density and shift |
| End-to-end correction | Initial/runtime calls, total tokens, service time and environment/ETA/checker/scheduler CPU time |

All numerical tables and PNG/SVG plots are produced from saved trajectories. Environment ticks, process CPU seconds and language-model service seconds remain distinct units.
