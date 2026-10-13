import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .executor import execute_episode
from .kitchen import Kitchen, SKILLS, validate_plan
from .method import METHODS, QuantileModel, calibration_offset, pinball
from .utils import load, plot_results, save, seed_all


SPLITS = {
    "train": ["cramped_room", "asymmetric_advantages"],
    "calibration": ["coordination_ring"],
    "validation": ["large_room"],
    "test": ["forced_coordination", "counter_circuit"],
}


def planner_prompt(kitchen, goal):
    return (
        "Create a causal DAG and preferred per-agent chains for "
        + str(goal)
        + ' valid soup deliveries. Return {vertices:[{id,agent:0|1,skill,station:[x,y],supply:[x,y],object,predecessors:[id],normal_duration}],chains:{"0":[id],"1":[id]}}. Allowed skills: '
        + str(SKILLS)
        + ". Only stage_onion needs supply. drop_object and pickup_object need object:onion|tomato|dish|soup. Use only listed coordinates. Include causal prerequisites for object transfers and pot reuse; preferred order is not a causal edge unless necessary. wait_pot waits for native cooking completion. Full fetch/put/start/wait/collect/serve work must be present. Later independent staging may be placed after a wait in the preferred chain without an artificial dependency edge. Counter handoffs are necessary if agents are separated. Respect native recipes and existing held objects.\nSTATIONS="
        + str(kitchen.stations())
        + "\nSTATE="
        + str(kitchen.state.to_dict())
    )


def make_plans(args):
    from .llm import LocalModel

    llm = LocalModel(args.model, args.revision, max_generated=18000, max_total=20000)
    episodes = []
    for layout in args.layouts or SPLITS[args.split]:
        if layout not in SPLITS[args.split]:
            raise ValueError("Layout outside the declared split")
        for index in range(args.count):
            seed = args.seed + index
            seed_all(seed)
            kitchen = Kitchen(layout, seed=seed)
            if args.random_start:
                kitchen.state = kitchen.mdp.get_random_start_state_fn(
                    random_start_pos=True
                )()
            if args.random_start:
                kitchen.state.objects.clear()
                for player in kitchen.state.players:
                    if player.has_object():
                        player.remove_object()
            llm.reset()
            plan = llm.json(planner_prompt(kitchen, args.goal_soups), "plan", 12000)
            validate_plan(plan, kitchen)
            if sum(v["skill"] == "serve" for v in plan["vertices"]) < args.goal_soups:
                raise ValueError("Plan omits required soup deliveries")
            floor = kitchen.mdp.get_valid_player_positions()
            blocked = list(floor[len(floor) // 2]) if args.controlled else None
            episodes.append(
                {
                    "id": layout + "_" + str(seed) + "_" + args.controller,
                    "layout": layout,
                    "seed": seed,
                    "split": args.split,
                    "controller": args.controller,
                    "controlled": args.controlled,
                    "action_duration": args.action_duration,
                    "blocked_cell": blocked,
                    "initial_state": kitchen.state.to_dict(),
                    "plan": plan,
                    "planner_ledger": llm.ledger.copy(),
                    "planner_model": args.model,
                    "planner_revision": args.revision,
                }
            )
            save(args.output, episodes)


def collect(args):
    cfg = load(args.config)
    records, episodes = [], []
    for episode in load(args.plans):
        result = execute_episode(episode, None, {}, cfg, "wait", "cpu")
        episodes.append(result)
        records.extend(result["wait_records"])
    if not records:
        raise ValueError("No dependency waits observed in supplied trajectories")
    save(args.output, records)
    save(Path(args.output).with_suffix(".episodes.json"), episodes)


def training_records(path, split):
    rows = [r for r in load(path) if r["split"] == split]
    if not rows:
        raise ValueError("No wait records for " + split)
    return rows


def train(args):
    cfg = load(args.config)
    seed_all(args.seed)
    training, validation = (
        training_records(args.train_records, "train"),
        training_records(args.validation_records, "validation"),
    )
    if {r["layout"] for r in training} & {r["layout"] for r in validation}:
        raise ValueError("Training and validation layouts overlap")
    x = torch.tensor([r["features"] for r in training], device=args.device)
    y = torch.tensor(
        [r["target"] for r in training], dtype=torch.float32, device=args.device
    )
    xv = torch.tensor([r["features"] for r in validation], device=args.device)
    yv = torch.tensor(
        [r["target"] for r in validation], dtype=torch.float32, device=args.device
    )
    model = QuantileModel(x.shape[1], cfg["horizon"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, history = float("inf"), []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["epochs"]):
        values = []
        model.train()
        for indices in torch.randperm(len(x), device=args.device).split(
            cfg["batch_size"]
        ):
            optimizer.zero_grad()
            loss = pinball(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation_loss = float(pinball(model(xv), yv))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(values)),
                "validation_pinball": validation_loss,
            }
        )
        if validation_loss < best:
            best = validation_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "inputs": x.shape[1],
                    "horizon": cfg["horizon"],
                    "config": cfg,
                    "seed": args.seed,
                },
                output / "eta.pt",
            )
        save(output / "history.json", history)
    means = defaultdict(list)
    for row in training:
        means[row["skill"]].append(row["target"])
    save(
        output / "type_means.json",
        {key: float(np.mean(values)) for key, values in means.items()},
    )


def model_from(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = QuantileModel(saved["inputs"], saved["horizon"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def one_per_episode(rows, seed=2027):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    rng = random.Random(seed)
    return [rng.choice(values) for key, values in sorted(grouped.items())]


def calibrate(args):
    rows = one_per_episode(training_records(args.records, "calibration"))
    model = model_from(args.checkpoint, args.device)
    with torch.inference_mode():
        predictions = (
            model(torch.tensor([r["features"] for r in rows], device=args.device))
            .cpu()
            .numpy()
        )
    target = np.asarray([r["target"] for r in rows], dtype=float)
    offset = calibration_offset(predictions, target, 0.1, model.horizon)
    save(
        args.output,
        {
            "offset": offset,
            "alpha": 0.1,
            "eligible_episodes": len(rows),
            "type_means": load(args.type_means),
        },
    )


def run(args):
    cfg = load(args.config)
    model = model_from(args.checkpoint, args.device)
    calibration = load(args.calibration)
    oracle = {}
    if args.oracle_traces:
        for episode in load(args.oracle_traces):
            oracle[episode["id"]] = {
                (r["owner"], r["head_id"]): r["release_tick"]
                for r in episode["release_events"]
            }
    if args.method == "oracle" and not oracle:
        raise ValueError("Oracle requires baseline release-event traces")
    results = []
    replanning_model = None
    if args.replan_model:
        from .llm import LocalModel

        replanning_model = LocalModel(
            args.replan_model,
            args.replan_revision,
            max_generated=cfg["planner_token_cap"],
            max_total=cfg["planner_token_cap"],
        )
    for episode in load(args.plans):
        if episode["split"] != args.split:
            continue
        seed_all(episode["seed"])
        replanner = None
        if replanning_model:
            replanning_model.reset()
            replanning_model.ledger = list(episode.get("planner_ledger", []))

            def replanner(kitchen, old_plan, reason):
                if len(replanning_model.ledger) >= cfg["planner_call_cap"]:
                    raise RuntimeError("Shared planner call cap exhausted")
                return replanning_model.json(
                    planner_prompt(kitchen, cfg["goal_soups"] - kitchen.soups)
                    + "\nINVALID_PLAN_REASON="
                    + reason
                    + "\nPREVIOUS_PLAN="
                    + str(old_plan),
                    "replan",
                    12000,
                )

        result = execute_episode(
            episode,
            model,
            calibration,
            cfg,
            args.method,
            args.device,
            oracle.get(episode["id"]),
            replanner,
        )
        result["runtime_planner_ledger"] = (
            replanning_model.ledger[len(episode.get("planner_ledger", [])) :]
            if replanning_model
            else []
        )
        result["planner_ledger"] = (
            list(result["planner_ledger"]) + result["runtime_planner_ledger"]
        )
        result["calibration"] = calibration
        result["config"] = cfg
        results.append(result)
        save(Path(args.output) / "traces.json", results)
        save(
            Path(args.output) / "wait_records.json",
            [record for item in results for record in item["wait_records"]],
        )


def evaluate(args):
    rows = [row for path in args.inputs for row in load(path)]
    model = model_from(args.checkpoint, args.device) if args.checkpoint else None
    summary = []
    for method, cohort in sorted({(r["method"], r["cohort"]) for r in rows}):
        episodes = [r for r in rows if r["method"] == method and r["cohort"] == cohort]
        categories = {
            key: sum(r["categories"][key] for r in episodes)
            for key in ("useful", "idle", "switch")
        }
        agent_ticks = sum(r["agent_ticks"] for r in episodes)
        insertions = [i for r in episodes for i in r["insertions"]]
        if sum(categories.values()) != agent_ticks:
            raise ValueError("Agent-tick categories do not partition elapsed execution")
        result = {
            "method": method + "_" + cohort,
            "cohort": cohort,
            "capped_completion_ticks": float(
                np.mean([r["completion_ticks"] for r in episodes])
            ),
            "success_percent": 100 * float(np.mean([r["success"] for r in episodes])),
            "soups_per_tick": sum(r["soups"] for r in episodes)
            / max(sum(r["completion_ticks"] for r in episodes), 1),
            "useful_share": categories["useful"] / max(agent_ticks, 1),
            "idle_share": categories["idle"] / max(agent_ticks, 1),
            "switch_share": categories["switch"] / max(agent_ticks, 1),
            "insertion_miss_rate": sum(r["missed"] for r in insertions)
            / len(insertions)
            if insertions
            else None,
            "attempted_insertions": len(insertions),
            "completed_insertions": sum(r["completed"] for r in insertions),
            "wall_seconds": float(np.mean([r["wall_seconds"] for r in episodes])),
            "missed_insertions": sum(r["missed"] for r in insertions),
            "planner_service_seconds": float(
                np.mean(
                    [sum(c["seconds"] for c in r["planner_ledger"]) for r in episodes]
                )
            ),
            "runtime_planner_calls": float(
                np.mean([len(r.get("runtime_planner_ledger", [])) for r in episodes])
            ),
            "planner_calls": float(
                np.mean([len(r["planner_ledger"]) for r in episodes])
            ),
            "planner_tokens": float(
                np.mean(
                    [
                        sum(
                            v["prompt_tokens"] + v["generated_tokens"]
                            for v in r["planner_ledger"]
                        )
                        for r in episodes
                    ]
                )
            ),
        }
        for stage in ("environment", "eta", "checking", "scheduling"):
            result[stage + "_cpu_seconds"] = float(
                np.mean([r["cpu_seconds"][stage] for r in episodes])
            )
        if model:
            records = one_per_episode([v for r in episodes for v in r["wait_records"]])
            if records:
                with torch.inference_mode():
                    prediction = (
                        model(
                            torch.tensor(
                                [r["features"] for r in records], device=args.device
                            )
                        )
                        .cpu()
                        .numpy()
                    )
                targets = np.asarray([r["target"] for r in records])
                offset = (
                    episodes[0]["calibration"]["offset"]
                    if method != "uncalibrated"
                    else 0
                )
                lower, upper = (
                    np.maximum(0, prediction[:, 0] - offset),
                    np.minimum(model.horizon, prediction[:, -1] + offset),
                )
                result["eligible_records"] = len(records)
                result["covered_records"] = int(
                    ((targets >= lower) & (targets <= upper)).sum()
                )
                result["eta_mae_ticks"] = float(
                    np.abs(prediction[:, 9] - targets).mean()
                )
                result["interval_coverage"] = float(
                    np.mean((targets >= lower) & (targets <= upper))
                )
                result["interval_width_ticks"] = float(np.mean(upper - lower))
        summary.append(result)
    save(args.output, {"summary": summary})


def main():
    parser = argparse.ArgumentParser(prog="slacksync")
    sub = parser.add_subparsers(dest="command", required=True)
    plans = sub.add_parser("make-plans")
    plans.add_argument("--split", choices=list(SPLITS), required=True)
    plans.add_argument("--layouts", nargs="+")
    plans.add_argument("--count", type=int, default=1)
    plans.add_argument("--seed", type=int, default=2027)
    plans.add_argument("--goal-soups", type=int, default=4)
    plans.add_argument(
        "--controller",
        choices=[
            "native",
            "cautious",
            "stochastic",
            "lagged",
            "lex",
            "random_tie",
            "clockwise",
            "counterclockwise",
            "reactive_yield",
            "persistent_reroute",
            "burst_stay",
        ],
        default="native",
    )
    plans.add_argument("--controlled", action="store_true")
    plans.add_argument("--action-duration", type=int, choices=[1, 2, 4], default=1)
    plans.add_argument("--random-start", action="store_true")
    plans.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    plans.add_argument("--revision", default="main")
    plans.add_argument("--output", required=True)
    variants = sub.add_parser("make-cohort")
    variants.add_argument("--plans", required=True)
    variants.add_argument(
        "--cohort",
        choices=[
            "native",
            "controlled",
            "extreme_shift",
            "congestion",
            "calibration_controller",
        ],
        required=True,
    )
    variants.add_argument("--output", required=True)
    collecting = sub.add_parser("collect")
    collecting.add_argument("--plans", required=True)
    collecting.add_argument("--config", default="configs/default.json")
    collecting.add_argument("--output", required=True)
    diagnostic_data = sub.add_parser("make-diagnostics")
    diagnostic_data.add_argument("--wait-records", required=True)
    diagnostic_data.add_argument("--count", type=int, default=120)
    diagnostic_data.add_argument("--seed", type=int, default=9001)
    diagnostic_data.add_argument("--output", required=True)
    diagnostic_run = sub.add_parser("run-diagnostics")
    diagnostic_run.add_argument("--input", required=True)
    diagnostic_run.add_argument("--checkpoint", required=True)
    diagnostic_run.add_argument("--calibration", required=True)
    diagnostic_run.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["slacksync", "wait", "fixed_eta"],
    )
    diagnostic_run.add_argument("--config", default="configs/default.json")
    diagnostic_run.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--train-records", required=True)
    training.add_argument("--validation-records", required=True)
    training.add_argument("--config", default="configs/default.json")
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--output", required=True)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--records", required=True)
    calibration.add_argument("--checkpoint", required=True)
    calibration.add_argument("--type-means", required=True)
    calibration.add_argument("--output", required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--plans", required=True)
    execution.add_argument("--checkpoint", required=True)
    execution.add_argument("--calibration", required=True)
    execution.add_argument("--method", choices=METHODS, default="slacksync")
    execution.add_argument("--split", choices=["validation", "test"], default="test")
    execution.add_argument("--oracle-traces")
    execution.add_argument("--replan-model")
    execution.add_argument("--replan-revision", default="main")
    execution.add_argument("--config", default="configs/default.json")
    execution.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--inputs", nargs="+", required=True)
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--output", required=True)
    for item in (training, calibration, execution, evaluation, diagnostic_run):
        item.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    plot = sub.add_parser("plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "make-plans":
        make_plans(args)
    elif args.command == "make-cohort":
        from .cohorts import make_cohort

        make_cohort(args.plans, args.output, args.cohort)
    elif args.command == "collect":
        collect(args)
    elif args.command == "make-diagnostics":
        from .diagnostics import make

        make(args.wait_records, args.output, args.count, args.seed)
    elif args.command == "run-diagnostics":
        from .diagnostics import run as run_diagnostics

        run_diagnostics(
            args.input,
            model_from(args.checkpoint, args.device),
            load(args.calibration),
            load(args.config),
            args.methods,
            args.device,
            args.output,
        )
    elif args.command == "train":
        train(args)
    elif args.command == "calibrate":
        calibrate(args)
    elif args.command == "run":
        run(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
