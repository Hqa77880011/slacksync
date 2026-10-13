from pathlib import Path

import numpy as np
import torch

from .method import ReleaseDistribution, choose, conflicts
from .utils import load, save


def make(records_path, output, count=120, seed=9001):
    rows = load(records_path)
    if not rows:
        raise ValueError(
            "Supply recorded dependency waits for diagnostic features and release targets"
        )
    rng = np.random.default_rng(seed)
    scenarios = []
    for density in (0.0, 0.25, 0.5, 0.75, 1.0):
        for shift in (0.5, 1.0, 2.0):
            for index in range(count):
                row = rows[int(rng.integers(len(rows)))]
                jobs = []
                for number in range(8):
                    duration = int(rng.integers(6, 18))
                    jobs.append(
                        {
                            "id": str(number),
                            "ready": number < int(8 * density),
                            "reads": [],
                            "writes": ["independent:" + str(number)],
                            "normal_duration": duration,
                            "execution": duration,
                            "outward": float(rng.integers(0, 4)),
                            "return": float(rng.integers(0, 4)),
                            "switch": 1.0,
                            "benefit": float(duration),
                            "rank": float(duration),
                            "congestion": 0.0,
                        }
                    )
                scenarios.append(
                    {
                        "id": f"{density}_{shift}_{index}",
                        "features": row["features"],
                        "skill": row["skill"],
                        "release_ticks": min(400.0, max(0.0, row["target"] * shift)),
                        "opportunity_density": density,
                        "timing_shift": shift,
                        "jobs": jobs,
                        "head_duration": 20,
                    }
                )
    save(output, scenarios)


def run(scenarios_path, model, calibration, cfg, methods, device, output):
    results = []
    for scenario in load(scenarios_path):
        with torch.inference_mode():
            quantiles = (
                model(torch.tensor(scenario["features"], device=device)[None])[0]
                .cpu()
                .numpy()
            )
        for method in methods:
            remaining = [dict(job) for job in scenario["jobs"]]
            elapsed, saved_work, misses, attempts, switching = 0.0, 0.0, 0, 0, 0.0
            while elapsed < scenario["release_ticks"]:
                available = [
                    j
                    for j in remaining
                    if j["ready"]
                    and not conflicts(
                        j, [{"reads": ["dependency"], "writes": ["dependency"]}]
                    )
                ]
                adjusted = np.clip(quantiles - elapsed, 0, cfg["horizon"])
                distribution = ReleaseDistribution(
                    adjusted,
                    0 if method == "uncalibrated" else calibration["offset"],
                    cfg["horizon"],
                )
                selected, _ = choose(
                    available,
                    distribution,
                    cfg,
                    method,
                    calibration["type_means"].get(scenario["skill"], 200),
                    max(0, scenario["release_ticks"] - elapsed)
                    if method == "oracle"
                    else None,
                )
                if selected is None:
                    break
                duration = (
                    selected["execution"]
                    + selected["outward"]
                    + selected["return"]
                    + selected["switch"]
                )
                elapsed += duration
                saved_work += selected["normal_duration"]
                switching += (
                    selected["outward"] + selected["return"] + selected["switch"]
                )
                attempts += 1
                misses += elapsed > scenario["release_ticks"]
                remaining = [job for job in remaining if job["id"] != selected["id"]]
            completion = (
                max(elapsed, scenario["release_ticks"])
                + scenario["head_duration"]
                + sum(j["normal_duration"] for j in remaining)
            )
            results.append(
                {
                    "id": scenario["id"],
                    "cohort": "synthetic",
                    "method": method,
                    "opportunity_density": scenario["opportunity_density"],
                    "timing_shift": scenario["timing_shift"],
                    "completion_ticks": min(completion, cfg["horizon"]),
                    "success": completion <= cfg["horizon"],
                    "attempted_insertions": attempts,
                    "missed_insertions": misses,
                    "saved_work_ticks": saved_work,
                    "switch_ticks": switching,
                    "eta_error": abs(quantiles[9] - scenario["release_ticks"]),
                    "covered": max(0, quantiles[0] - calibration["offset"])
                    <= scenario["release_ticks"]
                    <= min(cfg["horizon"], quantiles[-1] + calibration["offset"]),
                }
            )
    summary = []
    for method, density, shift in sorted(
        {(r["method"], r["opportunity_density"], r["timing_shift"]) for r in results}
    ):
        rows = [
            r
            for r in results
            if r["method"] == method
            and r["opportunity_density"] == density
            and r["timing_shift"] == shift
        ]
        attempts = sum(r["attempted_insertions"] for r in rows)
        summary.append(
            {
                "method": method + f"_shift{shift}",
                "opportunity_density": density,
                "timing_shift": shift,
                "capped_completion_ticks": float(
                    np.mean([r["completion_ticks"] for r in rows])
                ),
                "success_percent": 100 * float(np.mean([r["success"] for r in rows])),
                "insertion_miss_rate": sum(r["missed_insertions"] for r in rows)
                / attempts
                if attempts
                else None,
                "eta_mae_ticks": float(np.mean([r["eta_error"] for r in rows])),
                "interval_coverage": float(np.mean([r["covered"] for r in rows])),
            }
        )
    save(
        Path(output) / "results.json",
        {"summary": summary, "episodes": results, "cohort": "synthetic"},
    )
