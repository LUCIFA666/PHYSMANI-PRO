"""Paired closed-loop success metrics; no proxy forecasts are counted as grasps."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class EpisodeResult:
    method: str
    task: str
    raw_episode_id: str
    seed: int
    success: bool
    evaluation_kind: str
    checkpoint_sha256: str
    observation_contract_sha256: str
    collisions: int = 0
    replans: int = 0
    handovers: int = 0
    elapsed_seconds: float = 0.
    reaction_latencies_ms: tuple[float, ...] = ()
    target_position_error_m: float | None = None
    target_time_error_s: float | None = None
    terminal_relative_speed_m_s: float | None = None

    def __post_init__(self):
        if self.evaluation_kind != "closed_loop_simulator":
            raise ValueError("success table accepts only closed-loop simulator results")
        if type(self.success) is not bool or not self.checkpoint_sha256 or not self.observation_contract_sha256:
            raise ValueError("success and model/observation provenance are required")
        if self.collisions < 0 or self.replans < 0 or self.handovers < 0:
            raise ValueError("invalid episode counters")

    def key(self):
        return self.task, self.raw_episode_id, self.seed


def wilson(successes, count, z=1.959963984540054):
    if count <= 0:
        raise ValueError("empty evaluation")
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [float(max(0., center-radius)), float(min(1., center+radius))]


def compare(ours, baseline, *, expected_keys=None, bootstrap_samples=10000, seed=0):
    def index(rows):
        result = {r.key(): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate task/episode/seed")
        if len({r.method for r in rows}) > 1 or len({r.checkpoint_sha256 for r in rows}) > 1:
            raise ValueError("each comparison arm must use one method and one locked checkpoint")
        return result
    a, b = index(ours), index(baseline)
    if not a or set(a) != set(b):
        raise ValueError("paired comparison requires identical nonempty episode identities")
    if expected_keys is not None and set(a) != set(expected_keys):
        raise ValueError("evaluation incomplete against locked manifest")
    keys = sorted(a)
    for key in keys:
        if a[key].observation_contract_sha256 != b[key].observation_contract_sha256:
            raise ValueError("observation/control contract differs between methods")
    x, y = np.array([a[k].success for k in keys]), np.array([b[k].success for k in keys])
    differences = x.astype(float)-y.astype(float)
    rng = np.random.default_rng(seed)
    # Seeds of the same raw episode are correlated. Resample raw episode
    # clusters within each task, retaining every paired seed in that cluster.
    sample_sums = np.zeros(bootstrap_samples)
    sample_counts = np.zeros(bootstrap_samples)
    sample_ours, sample_baseline = np.zeros(bootstrap_samples), np.zeros(bootstrap_samples)
    raw_groups = {}
    for index, key in enumerate(keys):
        raw_groups.setdefault(key[:2], []).append(index)
    for task in sorted({k[0] for k in keys}):
        groups = [indices for (name, _), indices in raw_groups.items() if name==task]
        sums = np.array([differences[g].sum() for g in groups])
        ours_sums = np.array([x[g].sum() for g in groups])
        baseline_sums = np.array([y[g].sum() for g in groups])
        counts = np.array([len(g) for g in groups])
        choices = rng.integers(len(groups), size=(bootstrap_samples, len(groups)))
        sample_sums += sums[choices].sum(1)
        sample_counts += counts[choices].sum(1)
        sample_ours += ours_sums[choices].sum(1)
        sample_baseline += baseline_sums[choices].sum(1)
    sampled = sample_sums/sample_counts
    wins, losses = int(np.sum(x & ~y)), int(np.sum(~x & y))
    discordant = wins+losses
    tasks = {}
    for task in sorted({k[0] for k in keys}):
        selected = np.array([k[0] == task for k in keys])
        tasks[task] = {"episodes": int(selected.sum()), "ours_sr": float(x[selected].mean()),
                       "baseline_sr": float(y[selected].mean()), "paired_delta": float(differences[selected].mean())}
    repeated_seeds = len(raw_groups) != len(keys)
    return {"evaluation_kind": "closed_loop_simulator", "episodes": len(keys),
            "independent_raw_episodes": len(raw_groups),
            "bootstrap_unit": "task-stratified raw episode clusters, all paired seeds retained",
            "ours": {"successes": int(x.sum()), "sr": float(x.mean()),
                     "wilson_95": wilson(int(x.sum()), len(x)) if not repeated_seeds else None,
                     "raw_episode_cluster_bootstrap_95":np.quantile(sample_ours/sample_counts,[.025,.975]).tolist()},
            "baseline": {"successes": int(y.sum()), "sr": float(y.mean()),
                         "wilson_95": wilson(int(y.sum()), len(y)) if not repeated_seeds else None,
                         "raw_episode_cluster_bootstrap_95":np.quantile(sample_baseline/sample_counts,[.025,.975]).tolist()},
            "paired_delta": float(differences.mean()), "paired_bootstrap_95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "paired_wins": wins, "paired_losses": losses,
            "mcnemar_exact_p": (float(binomtest(wins, discordant, .5).pvalue) if discordant else 1.) if not repeated_seeds else None,
            "mcnemar_repeated_seed_limitation": "not computed because repeated seeds are correlated" if repeated_seeds else None,
            "wilson_intervals_assume_independent_trials": not repeated_seeds,
            "per_task": tasks, "macro_task_delta": float(np.mean([v["paired_delta"] for v in tasks.values()]))}


def append_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False)+"\n")
