import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


METHODS = (
    "slacksync",
    "wait",
    "every_wait_replan",
    "greedy",
    "shortest",
    "critical_path",
    "fixed_eta",
    "uncalibrated",
    "oracle",
    "no_return",
    "no_lateness",
    "no_congestion",
    "no_risk",
)
QUANTILES = np.arange(1, 20, dtype=float) / 20


class QuantileModel(nn.Module):
    def __init__(self, inputs, horizon=400):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 19),
        )
        self.horizon = horizon

    def forward(self, inputs):
        increments = F.softplus(self.network(inputs))
        return (
            self.horizon
            * increments.cumsum(-1)
            / (1 + increments.sum(-1, keepdim=True))
        )


def pinball(prediction, target):
    quantiles = prediction.new_tensor(QUANTILES)
    residual = target[:, None] - prediction
    return torch.maximum(quantiles * residual, (quantiles - 1) * residual).mean()


def calibration_offset(predictions, targets, alpha=0.1, horizon=400):
    residuals = np.maximum.reduce(
        [
            np.zeros(len(targets)),
            predictions[:, 0] - targets,
            targets - predictions[:, -1],
        ]
    )
    rank = math.ceil((len(residuals) + 1) * (1 - alpha))
    return (
        float(np.sort(residuals)[rank - 1])
        if rank <= len(residuals)
        else float(horizon)
    )


class ReleaseDistribution:
    def __init__(self, quantiles, offset=0, horizon=400):
        quantiles = np.asarray(quantiles, dtype=float).copy()
        if quantiles.shape != (19,) or np.any(np.diff(quantiles) < -1e-7):
            raise ValueError("Release quantiles must be nineteen ordered values")
        quantiles[QUANTILES < 0.5] -= offset
        quantiles[QUANTILES > 0.5] += offset
        self.times = np.r_[0.0, np.clip(quantiles, 0, horizon), float(horizon)]
        self.probabilities = np.r_[0.0, QUANTILES, 1.0]
        self.horizon = horizon

    def cdf(self, duration):
        if duration < 0:
            return 0.0
        if duration >= self.horizon:
            return 1.0
        index = int(np.searchsorted(self.times, duration, side="right") - 1)
        left, right = self.times[index : index + 2]
        low, high = self.probabilities[index : index + 2]
        return float(
            high
            if right == left
            else low + (high - low) * (duration - left) / (right - left)
        )

    def lateness(self, duration):
        duration = max(float(duration), 0.0)
        result = 0.0
        for i in range(len(self.times) - 1):
            left, right = self.times[i : i + 2]
            if left >= duration:
                break
            upper = min(right, duration)
            if upper > left:
                low, high = self.probabilities[i : i + 2]
                end = low + (high - low) * (upper - left) / (right - left)
                result += 0.5 * (low + end) * (upper - left)
        return float(result + max(0, duration - self.horizon))


def descendants(vertices, node):
    successors = {v["id"]: [] for v in vertices}
    for v in vertices:
        for predecessor in v["predecessors"]:
            successors[predecessor].append(v["id"])
    pending, result = [node], set()
    while pending:
        current = pending.pop()
        for successor in successors[current]:
            if successor not in result:
                result.add(successor)
                pending.append(successor)
    return result


def graph_ranks(vertices):
    indexed = {v["id"]: v for v in vertices}
    successors = {key: [] for key in indexed}
    for v in vertices:
        for parent in v["predecessors"]:
            if parent not in indexed:
                raise ValueError("Unknown causal predecessor: " + parent)
            successors[parent].append(v["id"])
    ranks, visiting = {}, set()

    def rank(key):
        if key in visiting:
            raise ValueError("Causal graph has a cycle")
        if key not in ranks:
            visiting.add(key)
            ranks[key] = indexed[key].get("normal_duration", 1) + max(
                (rank(c) for c in successors[key]), default=0
            )
            visiting.remove(key)
        return ranks[key]

    for key in indexed:
        rank(key)
    return ranks


def conflicts(candidate, active):
    reads, writes = set(candidate["reads"]), set(candidate["writes"])
    return any(
        writes & (set(job["reads"]) | set(job["writes"])) or reads & set(job["writes"])
        for job in active
    )


def choose(
    candidates,
    distribution,
    cfg,
    method="slacksync",
    fixed_eta=None,
    oracle_release=None,
):
    if method in ("wait", "every_wait_replan") or not candidates:
        return None, []
    ranked = []
    for candidate in candidates:
        duration = (
            candidate["execution"]
            + candidate["outward"]
            + candidate["return"]
            + candidate["switch"]
        )
        if method == "no_return":
            duration -= candidate["return"]
        lateness = distribution.lateness(duration)
        risk = distribution.cdf(duration)
        if method == "fixed_eta":
            lateness = max(0, duration - fixed_eta)
            risk = float(duration >= fixed_eta)
        if method == "oracle":
            if oracle_release is None:
                raise ValueError("Oracle needs evaluator release times")
            lateness = max(0, duration - oracle_release)
            risk = float(duration >= oracle_release)
        extra = max(0, duration - candidate["normal_duration"])
        value = (
            candidate["benefit"]
            - extra
            - (0 if method == "no_lateness" else cfg["lambda"]) * lateness
            - (0 if method == "no_congestion" else cfg["mu"]) * candidate["congestion"]
        )
        if method in ("greedy", "shortest", "critical_path"):
            risk = 0
            value = (
                1
                if method == "greedy"
                else 1 / max(duration, 1)
                if method == "shortest"
                else candidate["rank"] / max(duration, 1)
            )
        admissible = (
            method == "no_risk" or risk <= cfg["risk_threshold"]
        ) and value > 0
        ranked.append(
            dict(
                candidate,
                predicted_duration=duration,
                predicted_lateness=lateness,
                predicted_risk=risk,
                utility=value,
                admissible=admissible,
            )
        )
    allowed = [r for r in ranked if r["admissible"]]
    return max(allowed, key=lambda r: r["utility"]) if allowed else None, ranked
