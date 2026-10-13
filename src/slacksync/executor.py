import time

import numpy as np
import torch

from .kitchen import Kitchen, held, validate_plan
from .method import ReleaseDistribution, choose, conflicts, descendants, graph_ranks


def candidates(kitchen, vertices, done, owner, head, ranks, cfg):
    dependency = next((p for p in head["predecessors"] if p not in done), head["id"])
    downstream = descendants(vertices, dependency)
    indexed = {v["id"]: v for v in vertices}
    required = set(head["predecessors"]) - done
    pending = list(required)
    while pending:
        for predecessor in indexed[pending.pop()]["predecessors"]:
            if predecessor not in done and predecessor not in required:
                required.add(predecessor)
                pending.append(predecessor)
    active = [v for v in vertices if v["id"] in required]
    active.append({"reads": ["station:" + str(tuple(head["station"]))], "writes": []})
    player = kitchen.state.players[owner]
    resume = {
        "position": list(player.position),
        "orientation": list(player.orientation),
        "held": held(player),
    }
    possible = []
    rejected = []
    maximum = max(ranks.values())
    for job in vertices:
        if (
            job["agent"] != owner
            or job["id"] in done
            or job["id"] == head["id"]
            or job["id"] in downstream
        ):
            continue
        if not set(job["predecessors"]) <= done or not kitchen.ready(job):
            continue
        if (
            job["skill"] not in ("stage_onion", "start_cooking")
            or resume["held"] != "empty"
        ):
            rejected.append(
                {
                    "id": job["id"],
                    "reason": "Held-object resume state cannot be restored",
                }
            )
            continue
        if conflicts(job, active):
            rejected.append({"id": job["id"], "reason": "Dependency resource conflict"})
            continue
        first = job["supply"] if job["skill"] == "stage_onion" else job["station"]
        outward = kitchen.travel(owner, first)
        approaches = [p for p, _ in kitchen.approaches(job["station"])]
        returns = [kitchen.resume_route(p, resume) for p in approaches]
        returns = [r for r in returns if r is not None]
        if not returns or not np.isfinite(outward):
            rejected.append({"id": job["id"], "reason": "No outward or return path"})
            continue
        execution = 1.0
        if job["skill"] == "stage_onion":
            transit = [
                kitchen.path(p, approaches) for p, _ in kitchen.approaches(first)
            ]
            transit = [r for r in transit if r is not None]
            if not transit:
                continue
            execution = min(len(r) for r in transit) + 2.0
        return_cost = min(len(r) for r in returns) + 1.0
        congestion = sum(
            float(p.position in approaches)
            for i, p in enumerate(kitchen.state.players)
            if i != owner
        )
        possible.append(
            {
                "id": job["id"],
                "execution": execution,
                "outward": outward,
                "return": return_cost,
                "switch": 1.0,
                "normal_duration": job["normal_duration"],
                "benefit": job["normal_duration"] * ranks[job["id"]] / maximum,
                "rank": ranks[job["id"]],
                "congestion": congestion,
                "resume": resume,
            }
        )
    possible.sort(key=lambda c: c["rank"], reverse=True)
    return possible[: cfg["candidate_cap"]], rejected


def invalid_plan(kitchen, vertices, done, heads):
    for job in vertices:
        if job["id"] in done:
            continue
        if not np.isfinite(kitchen.travel(job["agent"], job["station"])):
            return "Assigned agent cannot reach required station: " + job["id"]
    indexed = {job["id"]: job for job in vertices}
    waits = {owner: set() for owner in range(2)}
    for owner, head in enumerate(heads):
        if head:
            waits[owner] = {
                indexed[key]["agent"] for key in head["predecessors"] if key not in done
            }
    if any(owner in waits[owner] for owner in waits) or (
        1 in waits[0] and 0 in waits[1]
    ):
        return "Preferred-chain dependency wait cycle"
    return None


def execute_episode(
    episode, model, calibration, cfg, method, device, oracle=None, replanner=None
):
    kitchen = Kitchen(
        episode["layout"],
        episode.get("initial_state"),
        episode.get("controlled", False),
        episode["seed"],
        episode.get("blocked_cell"),
        episode.get("action_duration", 1),
        episode.get("controller", "native"),
        episode.get("pause_exit", 0.5),
    )
    plan = validate_plan(episode["plan"], kitchen)
    vertices = plan["vertices"]
    jobs = {v["id"]: v for v in vertices}
    ranks = graph_ranks(vertices)
    done, running, waiting = set(), {}, {}
    records, insertions, ticks, decisions, rejected = [], [], [], [], []
    categories = {"useful": 0, "idle": 0, "switch": 0}
    times = {"environment": 0.0, "eta": 0.0, "checking": 0.0, "scheduling": 0.0}
    event_times = []
    corrections = []
    failure = None
    replan_waits = set()

    def head(owner):
        return next(
            (
                jobs[key]
                for key in plan["chains"][str(owner)]
                if key not in done
                and not (
                    owner in running
                    and running[owner]["insertion"]
                    and running[owner]["job"] == key
                )
            ),
            None,
        )

    start_wall = time.perf_counter()
    while kitchen.tick < cfg["horizon"] and kitchen.soups < cfg["goal_soups"]:
        if not running:
            invalid = invalid_plan(kitchen, vertices, done, [head(0), head(1)])
            if invalid:
                if replanner is None:
                    failure = invalid
                    break
                try:
                    replacement = replanner(kitchen, plan, invalid)
                    plan = validate_plan(replacement, kitchen)
                except (ValueError, RuntimeError, KeyError) as error:
                    failure = str(error)
                    break
                corrections.append(
                    {"tick": kitchen.tick, "reason": invalid, "plan": plan}
                )
                vertices = plan["vertices"]
                jobs = {v["id"]: v for v in vertices}
                ranks = graph_ranks(vertices)
                done = set()
        actions = [(0, 0), (0, 0)]
        tick_categories = ["idle", "idle"]
        for owner in range(2):
            current = head(owner)
            if current is None and owner not in running:
                continue
            if current is None:
                current = jobs[running[owner]["job"]]
            key = (owner, current["id"])
            prerequisite_ready = set(current["predecessors"]) <= done and kitchen.ready(
                current
            )
            if (
                key in waiting
                and prerequisite_ready
                and waiting[key]["release"] is None
            ):
                waiting[key]["release"] = kitchen.tick
                event_times.append(
                    {
                        "head_id": current["id"],
                        "owner": owner,
                        "release_tick": kitchen.tick,
                    }
                )
                for insertion in insertions:
                    if (
                        insertion["head_id"] == current["id"]
                        and insertion["owner"] == owner
                        and not insertion.get("release_observed")
                    ):
                        insertion["missed"] = not kitchen.resume_ready(
                            owner, insertion["resume"]
                        )
                        insertion["release_observed"] = True
            if owner in running:
                state = running[owner]
                job = jobs[state["job"]]
                if state.get("returning"):
                    if kitchen.resume_ready(owner, state["resume"]):
                        state["record"]["return_tick"] = kitchen.tick
                        del running[owner]
                    else:
                        actions[owner] = kitchen.resume_action(owner, state["resume"])
                        tick_categories[owner] = (
                            "switch" if actions[owner] != (0, 0) else "idle"
                        )
                        continue
                elif kitchen.complete(job, state["before"]):
                    done.add(job["id"])
                    if state["insertion"]:
                        state["returning"] = True
                        state["record"]["completed"] = True
                        state["record"]["completion_tick"] = kitchen.tick
                        actions[owner] = kitchen.resume_action(owner, state["resume"])
                        tick_categories[owner] = (
                            "switch" if actions[owner] != (0, 0) else "idle"
                        )
                        continue
                    del running[owner]
                else:
                    actions[owner] = kitchen.action(job)
                    tick_categories[owner] = (
                        "useful" if actions[owner] != (0, 0) else "idle"
                    )
                    continue
            current = head(owner)
            if current is None and owner not in running:
                continue
            if current is None:
                current = jobs[running[owner]["job"]]
            key = (owner, current["id"])
            if set(current["predecessors"]) <= done and kitchen.ready(current):
                if current["skill"] == "wait_pot":
                    done.add(current["id"])
                    continue
                running[owner] = {
                    "job": current["id"],
                    "before": kitchen.begin(current),
                    "insertion": False,
                }
                actions[owner] = kitchen.action(current)
                tick_categories[owner] = (
                    "useful" if actions[owner] != (0, 0) else "idle"
                )
                continue
            if method == "every_wait_replan" and key not in replan_waits:
                replan_waits.add(key)
                if replanner is None:
                    raise ValueError("every_wait_replan requires --replan-model")
                if not running:
                    try:
                        replacement = replanner(
                            kitchen, plan, "Dependency waiting: " + current["id"]
                        )
                        plan = validate_plan(replacement, kitchen)
                    except (ValueError, RuntimeError, KeyError) as error:
                        failure = str(error)
                        break
                    corrections.append(
                        {"tick": kitchen.tick, "reason": "wait", "plan": plan}
                    )
                    vertices = plan["vertices"]
                    jobs = {v["id"]: v for v in vertices}
                    ranks = graph_ranks(vertices)
                    done = set()
                    break
            if key not in waiting:
                waiting[key] = {
                    "start": kitchen.tick,
                    "release": None,
                    "samples": [],
                    "skill": current["skill"],
                }
            feature = kitchen.features(owner, current)
            waiting[key]["samples"].append(
                {"tick": kitchen.tick, "features": feature.tolist()}
            )
            start = time.perf_counter()
            available, denied = candidates(
                kitchen, vertices, done, owner, current, ranks, cfg
            )
            times["checking"] += time.perf_counter() - start
            rejected.extend(dict(r, tick=kitchen.tick, owner=owner) for r in denied)
            if model is None or method == "wait":
                quantiles = None
                chosen, scores = None, []
            else:
                start = time.perf_counter()
                with torch.inference_mode():
                    quantiles = (
                        model(torch.tensor(feature, device=device)[None])[0]
                        .cpu()
                        .numpy()
                    )
                times["eta"] += time.perf_counter() - start
                offset = 0 if method == "uncalibrated" else calibration.get("offset", 0)
                distribution = ReleaseDistribution(quantiles, offset, cfg["horizon"])
                start = time.perf_counter()
                future = None
                if oracle:
                    future = max(
                        0,
                        oracle.get((owner, current["id"]), cfg["horizon"])
                        - kitchen.tick,
                    )
                chosen, scores = choose(
                    available,
                    distribution,
                    cfg,
                    method,
                    calibration.get("type_means", {}).get(
                        current["skill"], cfg["horizon"] / 2
                    ),
                    future,
                )
                times["scheduling"] += time.perf_counter() - start
            decisions.append(
                {
                    "tick": kitchen.tick,
                    "owner": owner,
                    "head_id": current["id"],
                    "available": len(available),
                    "quantiles": quantiles.tolist() if quantiles is not None else None,
                    "chosen": chosen["id"] if chosen else None,
                    "scores": scores,
                }
            )
            if chosen:
                record = dict(
                    chosen,
                    owner=owner,
                    head_id=current["id"],
                    start_tick=kitchen.tick,
                    completed=False,
                    missed=False,
                    release_observed=False,
                )
                insertions.append(record)
                job = jobs[chosen["id"]]
                running[owner] = {
                    "job": job["id"],
                    "before": kitchen.begin(job),
                    "insertion": True,
                    "resume": chosen["resume"],
                    "record": record,
                }
                actions[owner] = kitchen.action(job)
                tick_categories[owner] = (
                    "useful" if actions[owner] != (0, 0) else "idle"
                )
        intended = list(actions)
        reserved_stations = {
            tuple(jobs[r["job"]]["station"]): owner
            for owner, r in running.items()
            if r["insertion"] and not r.get("returning")
        }
        for owner, action in enumerate(actions):
            if action != "interact" or owner not in running:
                continue
            job = jobs[running[owner]["job"]]
            target = tuple(
                np.asarray(kitchen.state.players[owner].position)
                + kitchen.state.players[owner].orientation
            )
            if not kitchen.valid_interaction(owner, job) or (
                target in reserved_stations and reserved_stations[target] != owner
            ):
                actions[owner] = (0, 0)
                tick_categories[owner] = "idle"
        positions_before = [p.position for p in kitchen.state.players]
        start = time.perf_counter()
        executed = kitchen.step(actions)
        times["environment"] += time.perf_counter() - start
        for owner in range(2):
            if executed[owner] == (0, 0):
                tick_categories[owner] = "idle"
            elif (
                isinstance(executed[owner], tuple)
                and kitchen.state.players[owner].position == positions_before[owner]
                and tuple(np.asarray(positions_before[owner]) + executed[owner])
                in kitchen.mdp.get_valid_player_positions()
            ):
                tick_categories[owner] = "idle"
            categories[tick_categories[owner]] += 1
        ticks.append(
            {
                "tick": kitchen.tick,
                "actions": [list(a) if isinstance(a, tuple) else a for a in executed],
                "intended": [list(a) if isinstance(a, tuple) else a for a in intended],
                "categories": tick_categories,
                "soups": kitchen.soups,
            }
        )
    for (owner, identifier), wait in waiting.items():
        release = wait["release"] if wait["release"] is not None else cfg["horizon"]
        for sample in wait["samples"]:
            target = (
                min(cfg["horizon"], max(0, release - sample["tick"]))
                if wait["release"] is not None
                else cfg["horizon"]
            )
            records.append(
                {
                    "episode_id": episode["id"],
                    "layout": episode["layout"],
                    "split": episode["split"],
                    "seed": episode["seed"],
                    "controller": episode.get("controller", "native"),
                    "skill": wait["skill"],
                    "owner": owner,
                    "head_id": identifier,
                    "features": sample["features"],
                    "target": target,
                    "tick": sample["tick"],
                    "released": wait["release"] is not None,
                }
            )
    for record in insertions:
        if not record.get("release_observed"):
            record["missed"] = (
                True
                if record.get("return_tick", cfg["horizon"] + 1) > cfg["horizon"]
                else False
            )
    horizon = cfg["horizon"]
    result = {
        "id": episode["id"],
        "layout": episode["layout"],
        "controller": episode.get("controller", "native"),
        "seed": episode["seed"],
        "method": method,
        "cohort": episode.get(
            "cohort", "controlled" if episode.get("controlled") else "native"
        ),
        "success": kitchen.soups >= cfg["goal_soups"],
        "completion_ticks": kitchen.tick
        if kitchen.soups >= cfg["goal_soups"]
        else horizon,
        "soups": kitchen.soups,
        "agent_ticks": 2 * kitchen.tick,
        "categories": categories,
        "insertions": insertions,
        "decisions": decisions,
        "rejected": rejected,
        "wait_records": records,
        "release_events": event_times,
        "ticks": ticks,
        "cpu_seconds": times,
        "wall_seconds": time.perf_counter() - start_wall,
        "planner_ledger": episode.get("planner_ledger", []),
    }
    result["plan_corrections"] = corrections
    result["execution_failure"] = failure
    return result
