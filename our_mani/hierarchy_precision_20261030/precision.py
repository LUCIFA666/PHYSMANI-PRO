"""Physical-metre position objectives and fixed-cohort precision reporting.

This module performs no file reads, model loading, training or source selection.
Candidate handoff positions and commanded TCP positions are distinct metrics.
Every planned sample must be accounted for; unknown/nonfinite observations may
never improve an average by disappearing from the denominator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

import numpy as np


SCHEMA = "hierarchy_physical_position_precision_20261030_v1"
MEAN_TARGET_M = .03
HARD_LIMIT_M = .05
HEADS = ("candidate", "tcp")
SPLITS = ("train", "validation", "test")
USAGES = ("training", "selection", "evaluation")


def validate_split_usage(split, usage):
    """Reject TEST for fitting or checkpoint selection, before position access."""
    if not isinstance(split, str) or not isinstance(usage, str):
        raise ValueError("explicit split and usage strings are required")
    split, usage = split.lower(), usage.lower()
    if split == "val":
        split = "validation"
    if split not in SPLITS or usage not in USAGES:
        raise ValueError("split must be train/validation/test and usage training/selection/evaluation")
    if usage == "training" and split != "train":
        raise ValueError("only TRAIN samples may contribute to training")
    if usage == "selection" and split == "test":
        raise ValueError("TEST samples cannot contribute to model selection")
    return split, usage


def position_loss_m(predicted_xyz, target_xyz, *, split, usage="training", reduction="mean"):
    """Euclidean position error in metres, with an optional differentiable mean.

    Inputs are [...,3] coordinates in the same declared physical frame and metre
    units. There is no workspace scaling, squaring (m²), clipping, mask or sample
    exclusion. Nonfinite, empty and mismatched inputs fail instead of becoming a
    zero loss. Torch tensors retain their gradient path; NumPy/list inputs return
    NumPy values. ``split`` is mandatory so TEST cannot silently enter fitting.
    """
    validate_split_usage(split, usage)
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be none, mean or sum")
    # Lazy torch import keeps report-only usage independent of torch/GPU setup.
    try:
        import torch
    except ImportError:
        torch = None
    tensor = next((value for value in (predicted_xyz, target_xyz)
                   if torch is not None and isinstance(value, torch.Tensor)), None)
    if tensor is not None:
        if not tensor.is_floating_point():
            raise ValueError("physical position tensors must use a floating dtype")
        predicted = torch.as_tensor(predicted_xyz, dtype=tensor.dtype, device=tensor.device)
        target = torch.as_tensor(target_xyz, dtype=tensor.dtype, device=tensor.device)
        if (predicted.shape != target.shape or predicted.ndim < 1
                or predicted.shape[-1] != 3 or predicted.numel() == 0):
            raise ValueError("nonempty aligned [...,3] physical positions required")
        if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError("nonfinite physical positions cannot enter the objective")
        distances = torch.linalg.vector_norm(predicted - target, dim=-1)
        return distances if reduction == "none" else distances.mean() if reduction == "mean" else distances.sum()
    predicted, target = np.asarray(predicted_xyz, dtype=np.float64), np.asarray(target_xyz, dtype=np.float64)
    if (predicted.shape != target.shape or predicted.ndim < 1
            or predicted.shape[-1] != 3 or predicted.size == 0):
        raise ValueError("nonempty aligned [...,3] physical positions required")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite physical positions cannot enter the objective")
    distances = np.linalg.norm(predicted - target, axis=-1)
    return distances if reduction == "none" else distances.mean() if reduction == "mean" else distances.sum()


def _sample_key(row, usage):
    if not isinstance(row, Mapping):
        raise ValueError("sample and cohort entries must be mappings")
    split, _ = validate_split_usage(row["split"], usage)
    task, component, sample_id = row["task"], row["head"], row["sample_id"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("nonempty explicit task identity required")
    if not isinstance(component, str) or component.lower() not in HEADS:
        raise ValueError("head must explicitly identify candidate or tcp")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("stable nonempty sample_id required")
    return split, task, component.lower(), sample_id


def _key_record(key):
    return dict(zip(("split", "task", "head", "sample_id"), key))


def _sample_error(row):
    values = []
    for name in ("pred_xyz", "target_xyz"):
        if name not in row or row[name] is None:
            return None, "missing_" + name
        try:
            value = np.asarray(row[name], dtype=np.float64)
        except (TypeError, ValueError):
            return None, "invalid_" + name
        if value.shape != (3,):
            return None, "invalid_shape_" + name
        if not np.isfinite(value).all():
            return None, "nonfinite_" + name
        values.append(value)
    error = float(np.linalg.norm(values[0] - values[1]))
    if not np.isfinite(error):
        return None, "nonfinite_position_error"
    return error, None


def _group_metrics(keys, errors, invalid, *, unplanned_count):
    total = len(keys)
    valid_errors = [errors[key] for key in keys if key in errors]
    invalid_count = sum(key in invalid for key in keys)
    complete = invalid_count == 0 and unplanned_count == 0 and len(valid_errors) == total and total > 0
    stats = dict(mean_m=None, p50_m=None, p95_m=None, max_m=None)
    if complete:
        # These are all planned samples, never a filtered finite-only subset.
        values = np.asarray(valid_errors, dtype=np.float64)
        stats = dict(mean_m=float(values.mean()), p50_m=float(np.quantile(values, .5)),
                     p95_m=float(np.quantile(values, .95)), max_m=float(values.max()))
    within_3 = sum(value <= MEAN_TARGET_M for value in valid_errors)
    within_5 = sum(value <= HARD_LIMIT_M for value in valid_errors)
    hard_gate = complete and all(value <= HARD_LIMIT_M for value in valid_errors)
    mean_target = complete and stats["mean_m"] <= MEAN_TARGET_M
    return dict(expected_samples=total, valid_samples=len(valid_errors), invalid_or_missing_samples=invalid_count,
                unplanned_samples=unplanned_count, cohort_complete=complete, **stats,
                samples_le_3cm=within_3, samples_le_5cm=within_5,
                fraction_le_3cm=within_3 / total if total and unplanned_count == 0 else None,
                fraction_le_5cm=within_5 / total if total and unplanned_count == 0 else None,
                all_samples_le_5cm=bool(hard_gate), mean_le_3cm=bool(mean_target),
                hard_gate_passed=bool(hard_gate), precision_goal_met=bool(hard_gate and mean_target))


def precision_report(samples, *, expected_samples, usage="evaluation"):
    """Report a complete planned cohort separately by split, task and component.

    Each expected ledger row requires ``split``, ``task``, ``head`` and a
    stable ``sample_id``. Each observed row adds ``pred_xyz``/``target_xyz``
    (metres). Entirely missing rows, duplicate observations, extra unplanned rows
    and nonfinite/malformed positions all fail closed. If a group is incomplete,
    its mean/percentiles/max are None rather than a misleading filtered mean.
    Invalid/missing planned rows remain in the <=3cm/<=5cm ratio denominator.

    Evaluation may include TEST facts. Training or selection usage rejects TEST
    ledger entries before any observed position data is read. No statistic pools
    TRAIN, validation and TEST, or candidate and TCP, into one averaged score.
    """
    if usage not in USAGES:
        raise ValueError("explicit valid report usage required")
    expected = list(expected_samples)
    if not expected:
        raise ValueError("a nonempty complete expected-sample ledger is mandatory")
    expected_keys = [_sample_key(row, usage) for row in expected]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("expected-sample ledger contains duplicate identities")
    planned = set(expected_keys)
    observed = defaultdict(list)
    observed_count = 0
    for row in samples:
        key = _sample_key(row, usage)
        observed[key].append(row)
        observed_count += 1
    unplanned = {key: rows for key, rows in observed.items() if key not in planned}
    errors, invalid = {}, {}
    for key in expected_keys:
        rows = observed.get(key, [])
        if not rows:
            invalid[key] = "missing_sample"
        elif len(rows) != 1:
            invalid[key] = "duplicate_observed_sample"
        else:
            error, reason = _sample_error(rows[0])
            if reason is None:
                errors[key] = error
            else:
                invalid[key] = reason
    groups = defaultdict(list)
    for key in expected_keys:
        groups[key[:3]].append(key)
    unplanned_groups = Counter(key[:3] for key in unplanned)
    by_split = {}
    for (split, task, component), keys in sorted(groups.items()):
        split_report = by_split.setdefault(split, dict(tasks={}, hard_gate_passed=True,
                                                       mean_target_met=True, precision_goal_met=True))
        task_report = split_report["tasks"].setdefault(task, {})
        metrics = _group_metrics(keys, errors, invalid, unplanned_count=unplanned_groups[(split, task, component)])
        task_report[component] = metrics
        split_report["hard_gate_passed"] &= metrics["hard_gate_passed"]
        split_report["mean_target_met"] &= metrics["mean_le_3cm"]
        split_report["precision_goal_met"] &= metrics["precision_goal_met"]
    for key in unplanned:
        split = key[0]
        split_report = by_split.setdefault(split, dict(tasks={}, hard_gate_passed=False,
                                                       mean_target_met=False, precision_goal_met=False))
        split_report.update(hard_gate_passed=False, mean_target_met=False, precision_goal_met=False)
    return dict(schema=SCHEMA, usage=usage, position_units="metres",
                thresholds_m=dict(mean_target=MEAN_TARGET_M, all_samples_hard_limit=HARD_LIMIT_M),
                hard_gate_definition="every planned sample is present once, finite, and has Euclidean error <=0.05m; never a p95 substitution",
                mean_target_definition="each split/task/component mean error <=0.03m, with a complete finite cohort",
                expected_samples=len(expected_keys), observed_rows=observed_count,
                invalid_samples=[dict(**_key_record(key), reason=reason) for key, reason in sorted(invalid.items())],
                unplanned_samples=[dict(**_key_record(key), observed_rows=len(rows)) for key, rows in sorted(unplanned.items())],
                duplicate_observed_keys=[dict(**_key_record(key), observed_rows=len(rows))
                                         for key, rows in sorted(observed.items()) if len(rows) > 1],
                no_samples_filtered=True, splits_never_pooled=True,
                candidate_and_tcp_never_pooled=True,
                TEST_permitted_for_training_or_selection=False, by_split=by_split)
