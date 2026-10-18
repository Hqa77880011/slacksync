# SlackSync

Implementation of **SlackSync: Opportunity-Cost-Aware Work Insertion During Multi-Agent Dependency Waiting**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation

Install the supported [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) revision. Plan generation uses a local Hugging Face model; generated plans and simulator trajectories are not human gameplay recordings. Layout splits are fixed in the implementation.

```bash
python -m pip install "overcooked-ai @ git+https://github.com/HumanCompatibleAI/overcooked_ai@739950a079cdaed5a44fcc662efc40244c205d06"
slacksync make-plans --split train --count 1000 --seed 2027 --random-start --output data/train_plans.json
slacksync make-plans --split calibration --count 400 --seed 2027 --random-start --output data/calibration_plans.json
slacksync make-plans --split validation --count 300 --seed 2027 --random-start --output data/validation_plans.json
slacksync collect --plans data/train_plans.json --output data/train_waits.json
slacksync collect --plans data/calibration_plans.json --output data/calibration_waits.json
slacksync collect --plans data/validation_plans.json --output data/validation_waits.json
```

## Training and calibration

Keep model seed 17 and the configuration fixed across fitting and evaluation. Calibration uses its separate layout cohort.

```bash
slacksync train --train-records data/train_waits.json --validation-records data/validation_waits.json --seed 17 --output checkpoints/eta
slacksync calibrate --records data/calibration_waits.json --checkpoint checkpoints/eta/eta.pt --type-means checkpoints/eta/type_means.json --output checkpoints/calibration.json
```

## Execution and evaluation

```bash
slacksync make-plans --split test --count 150 --seed 2027 --random-start --output data/test_plans.json
slacksync run --plans data/test_plans.json --checkpoint checkpoints/eta/eta.pt --calibration checkpoints/calibration.json --method slacksync --output runs/slacksync
slacksync evaluate --inputs runs/slacksync/traces.json --checkpoint checkpoints/eta/eta.pt --output runs/comparison.json
slacksync plot --input runs/comparison.json --output runs/figures
```

## Comparisons and ablations

Reuse the same plans/checkpoints with methods `wait`, `fixed_eta`, `uncalibrated`, `no_return`, `no_lateness`, `no_congestion` or `no_risk`, then evaluate their trace files together.

```bash
slacksync run --plans data/test_plans.json --checkpoint checkpoints/eta/eta.pt --calibration checkpoints/calibration.json --method wait --output runs/wait
slacksync evaluate --inputs runs/slacksync/traces.json runs/wait/traces.json --checkpoint checkpoints/eta/eta.pt --output runs/comparison.json
```

## Outputs

Plan JSONs retain layout, seed, initial state and planner ledger. Collection writes wait records; fitting writes `eta.pt`, `history.json` and `type_means.json`; calibration writes the requested JSON. Runs contain `traces.json` and `wait_records.json`. Evaluation writes aggregate metrics; figures are PNG/SVG.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
