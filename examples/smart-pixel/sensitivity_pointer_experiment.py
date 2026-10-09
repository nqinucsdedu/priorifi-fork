"""Sensitivity-pointer grid experiment for smart-pixel PrioriFI.

This script runs a dedicated experiment that compares sensitivity-pointer strategies
across aggregation mode (mean/median) and history window size (last_k) without
modifying the existing fault injection campaign behavior.

Example:
    python sensitivity_pointer_experiment.py \
      --config ./dense_baseline_fkeras/fkeras_dense_model_58.yaml \
      --pretrained-model ./dense_baseline_fkeras/fkeras_dense_model_58.h5 \
      --model_id dense_baseline \
      --output_dir ./experiment_outputs \
      --aggregation_modes mean,median \
      --last_k_values 1,3,5 \
      --fic_range_start 0 --fic_range_stop 10000 --fic_range_step 1 \
      --bit_width 16

Expected outputs in --output_dir:
  - sensitivity_pointer_experiment_<timestamp>.json
  - sensitivity_pointer_experiment_<timestamp>.csv
  - sensitivity_pointer_experiment_heatmap_<timestamp>.png (if plotting succeeds)

Summary score:
  `priority_weighted_metric` rewards configurations that surface higher metric
  values earlier in the campaign. Larger scores indicate better prioritization.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml


SUPPORTED_AGGREGATIONS = ("mean", "median")


@dataclass(frozen=True)
class ExperimentConfig:
    aggregation_mode: str
    last_k: int


def _parse_list_argument(value: str) -> List[Any]:
    """Safely parse a list-like CLI value (JSON/Python literal list or CSV string)."""
    raw = value.strip()
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list literal, got: {value}")
        return parsed

    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_aggregation_modes(value: str) -> List[str]:
    modes = [str(mode).strip().lower() for mode in _parse_list_argument(value)]
    if not modes:
        raise ValueError("--aggregation_modes must include at least one mode")
    invalid = sorted(set(mode for mode in modes if mode not in SUPPORTED_AGGREGATIONS))
    if invalid:
        raise ValueError(
            f"Unsupported aggregation mode(s): {invalid}. Supported: {SUPPORTED_AGGREGATIONS}"
        )
    return modes


def parse_last_k_values(value: str) -> List[int]:
    raw_values = _parse_list_argument(value)
    if not raw_values:
        raise ValueError("--last_k_values must include at least one value")

    parsed: List[int] = []
    for raw in raw_values:
        try:
            parsed.append(int(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid last_k value: {raw}") from exc

    if any(last_k <= 0 for last_k in parsed):
        raise ValueError("All --last_k_values entries must be positive integers")
    return parsed


def parse_layer_precision_info(value: str | None) -> List[Tuple[int, int]] | None:
    if value is None:
        return None

    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("--layer_precision_info must be a list of (param_count, bit_width) tuples")

    normalized: List[Tuple[int, int]] = []
    for item in parsed:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise ValueError(
                "--layer_precision_info entries must be tuples of (param_count, bit_width)"
            )
        param_count, bit_width = item
        if not (isinstance(param_count, int) and isinstance(bit_width, int)):
            raise ValueError("--layer_precision_info tuple values must be integers")
        if param_count <= 0 or bit_width <= 0:
            raise ValueError("--layer_precision_info tuple values must be positive")
        normalized.append((param_count, bit_width))

    return normalized


def validate_fic_range(start: int, stop: int, step: int, max_bits: int) -> Tuple[int, int, int]:
    if step <= 0:
        raise ValueError("--fic_range_step must be a positive integer")
    if start < 0:
        raise ValueError("--fic_range_start must be >= 0")
    if stop > max_bits:
        raise ValueError(
            f"--fic_range_stop ({stop}) exceeds number of model bits ({max_bits})"
        )
    if start >= stop:
        raise ValueError("--fic_range_start must be less than --fic_range_stop")
    return start, stop, step


def compute_sensitivity_pointer(
    wbi_lists: Sequence[Sequence[int]],
    wbi_list_delta_metrics: Sequence[Sequence[float]],
    last_k: int,
    aggregation_mode: str,
) -> int:
    """Select next list index using configurable aggregation and deterministic ties."""
    if last_k <= 0:
        raise ValueError("last_k must be positive")

    mode = aggregation_mode.lower()
    if mode not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation mode: {aggregation_mode}. Supported: {SUPPORTED_AGGREGATIONS}"
        )

    best_idx: int | None = None
    best_value = float("-inf")

    for idx, bit_list in enumerate(wbi_lists):
        if len(bit_list) == 0:
            continue

        history = list(wbi_list_delta_metrics[idx])[-last_k:]
        if not history:
            agg_value = float("-inf")
        elif mode == "mean":
            agg_value = float(mean(history))
        else:  # mode == "median"
            agg_value = float(median(history))

        if (agg_value > best_value) or (
            agg_value == best_value and (best_idx is None or idx < best_idx)
        ):
            best_idx = idx
            best_value = agg_value

    if best_idx is None:
        raise ValueError("No non-empty bit list available for sensitivity pointer selection")

    return best_idx


def filter_bit_lists_by_range(
    bit_lists: Sequence[Sequence[int]],
    fic_range: Tuple[int, int, int],
) -> List[List[int]]:
    allowed_bits = set(range(*fic_range))
    return [[bit_idx for bit_idx in bit_list if bit_idx in allowed_bits] for bit_list in bit_lists]


def priority_weighted_metric(metric_values: Sequence[float]) -> float:
    """Order-sensitive score: larger is better (higher metrics found earlier)."""
    if not metric_values:
        return 0.0

    n = len(metric_values)
    numerator = 0.0
    denominator = 0.0
    for idx, value in enumerate(metric_values):
        weight = float(n - idx)
        numerator += weight * float(value)
        denominator += weight
    return 0.0 if denominator == 0.0 else numerator / denominator


def _flip_and_measure(
    fmodel: Any,
    bit_idx: int,
    x_test: Any,
    y_pred_reference: Any,
    eval_metric_func: Any,
) -> float:
    fmodel.explicitly_flip_bits([bit_idx])
    y_pred_fault = fmodel.model.predict(x_test, verbose=0)
    metric_value = float(eval_metric_func(y_pred_reference, y_pred_fault))
    fmodel.explicitly_reset_bits([bit_idx])
    return metric_value


def run_single_experiment(
    campaign_module: Any,
    fk_module: Any,
    load_model_tuple: Tuple[Dict[str, Any], str | None],
    x_test_pred_correct: np.ndarray,
    hess_ranking: Sequence[int],
    bit_width: int | None,
    layer_precision_info: List[Tuple[int, int]] | None,
    fic_range: Tuple[int, int, int],
    experiment_config: ExperimentConfig,
) -> Dict[str, Any]:
    config, pretrained_model = load_model_tuple
    model = campaign_module.load_model(config, pretrained_model=pretrained_model)
    fmodel = fk_module.fmodel.FModelAlt(model, incl_biases=True)

    y_pred = fmodel.model.predict(x_test_pred_correct, verbose=0)
    metric_baseline = float(campaign_module.my_eval_metric_00(y_pred, y_pred))

    bit_lists = campaign_module.convert_params_into_bit_lists(
        hess_ranking,
        layer_precision_info=layer_precision_info,
        bits_per_weight=bit_width,
    )
    bit_lists = filter_bit_lists_by_range(bit_lists, fic_range)
    metric_histories: List[List[float]] = [[] for _ in range(len(bit_lists))]

    metric_records: List[Tuple[int, float]] = [(-1, metric_baseline)]
    alerts: List[int] = []

    start_time = time.time()

    num_flips = 0
    for list_idx, bit_list in enumerate(bit_lists):
        if not bit_list:
            continue
        bit_idx = bit_list.pop(0)
        metric_value = _flip_and_measure(
            fmodel=fmodel,
            bit_idx=bit_idx,
            x_test=x_test_pred_correct,
            y_pred_reference=y_pred,
            eval_metric_func=campaign_module.my_eval_metric_00,
        )
        metric_histories[list_idx].append(metric_value)
        metric_records.append((bit_idx, metric_value))
        if campaign_module.my_alert_func_00(metric_baseline, metric_value):
            alerts.append(bit_idx)
        num_flips += 1

    while any(bit_list for bit_list in bit_lists):
        pointer_idx = compute_sensitivity_pointer(
            bit_lists,
            metric_histories,
            last_k=experiment_config.last_k,
            aggregation_mode=experiment_config.aggregation_mode,
        )
        bit_idx = bit_lists[pointer_idx].pop(0)
        metric_value = _flip_and_measure(
            fmodel=fmodel,
            bit_idx=bit_idx,
            x_test=x_test_pred_correct,
            y_pred_reference=y_pred,
            eval_metric_func=campaign_module.my_eval_metric_00,
        )
        metric_histories[pointer_idx].append(metric_value)
        metric_records.append((bit_idx, metric_value))
        if campaign_module.my_alert_func_00(metric_baseline, metric_value):
            alerts.append(bit_idx)
        num_flips += 1

    elapsed_seconds = time.time() - start_time
    faulty_metric_values = [value for _, value in metric_records[1:]]

    return {
        "aggregation_mode": experiment_config.aggregation_mode,
        "last_k": experiment_config.last_k,
        "elapsed_seconds": elapsed_seconds,
        "num_flips": num_flips,
        "baseline_metric": metric_baseline,
        "metric_values": faulty_metric_values,
        "alerts": alerts,
        "alert_count": len(alerts),
        "priority_weighted_metric": priority_weighted_metric(faulty_metric_values),
        "score_description": (
            "Priority-weighted metric (higher is better): weighted average of per-flip "
            "metric values with larger weights on earlier flips."
        ),
    }


def save_results(
    output_dir: str,
    timestamp: str,
    results: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"sensitivity_pointer_experiment_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    csv_path = os.path.join(output_dir, f"sensitivity_pointer_experiment_{timestamp}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "aggregation_mode",
            "last_k",
            "elapsed_seconds",
            "num_flips",
            "baseline_metric",
            "alert_count",
            "priority_weighted_metric",
            "metric_values",
            "alerts",
            "score_description",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            flattened = dict(row)
            flattened["metric_values"] = json.dumps(flattened["metric_values"])
            flattened["alerts"] = json.dumps(flattened["alerts"])
            writer.writerow(flattened)

    return json_path, csv_path


def save_heatmap(
    output_dir: str,
    timestamp: str,
    results: Sequence[Dict[str, Any]],
    aggregation_modes: Sequence[str],
    last_k_values: Sequence[int],
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not available; skipping heat-map generation.")
        return None

    score_map = {
        (result["aggregation_mode"], result["last_k"]): result["priority_weighted_metric"]
        for result in results
    }

    data = [[0.0 for _ in last_k_values] for _ in aggregation_modes]
    for i, mode in enumerate(aggregation_modes):
        for j, last_k in enumerate(last_k_values):
            data[i][j] = score_map[(mode, last_k)]

    fig, ax = plt.subplots(figsize=(1.5 * len(last_k_values) + 4, 4 + 0.3 * len(aggregation_modes)))

    try:
        import seaborn as sns

        sns.heatmap(
            data,
            annot=True,
            fmt=".4f",
            cmap="viridis",
            xticklabels=[str(v) for v in last_k_values],
            yticklabels=aggregation_modes,
            ax=ax,
        )
    except ImportError:
        heat = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_xticks(list(range(len(last_k_values))))
        ax.set_xticklabels([str(v) for v in last_k_values])
        ax.set_yticks(list(range(len(aggregation_modes))))
        ax.set_yticklabels(list(aggregation_modes))
        fig.colorbar(heat, ax=ax)

    ax.set_xlabel("last_k")
    ax.set_ylabel("aggregation_mode")
    ax.set_title("Sensitivity-pointer strategy comparison")

    heatmap_path = os.path.join(
        output_dir, f"sensitivity_pointer_experiment_heatmap_{timestamp}.png"
    )
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    return heatmap_path


def _build_experiment_grid(
    aggregation_modes: Iterable[str],
    last_k_values: Iterable[int],
) -> List[ExperimentConfig]:
    grid: List[ExperimentConfig] = []
    for mode in aggregation_modes:
        for last_k in last_k_values:
            grid.append(ExperimentConfig(aggregation_mode=mode, last_k=last_k))
    return grid


def main(args: argparse.Namespace) -> None:
    import fkeras as fk
    import tensorflow as tf
    from fkeras.metrics.hessian import HessianMetrics

    import fault_injection_campaign as campaign

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    aggregation_modes = parse_aggregation_modes(args.aggregation_modes)
    last_k_values = parse_last_k_values(args.last_k_values)
    layer_precision_info = parse_layer_precision_info(args.layer_precision_info)

    model = campaign.load_model(config, pretrained_model=args.pretrained_model)

    X_train, y_train, X_test, y_test = campaign.load_data()
    assert X_train.shape == (45415, 13)
    assert X_test.shape == (11113, 13)
    assert y_train.shape == (45415, 1)
    assert y_test.shape == (11113, 1)

    x_test_pred_correct, _ = campaign.gen_smart_pix_0mispredicts_dataset(model, X_test, y_test)

    hess = HessianMetrics(
        model,
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        X_test,
        y_test,
        batch_size=1024,
    )
    eigenvalues, eigenvectors = hess.top_k_eigenvalues(k=8, max_iter=500, rank_BN=False)
    hess_ranking, _ = hess.hessian_ranking_general(eigenvectors, eigenvalues=eigenvalues, k=8)

    nmpb = fk.fmodel.FModelAlt(model, incl_biases=True).num_model_param_bits
    fic_range_start = 0 if args.fic_range_start is None else args.fic_range_start
    fic_range_stop = nmpb if args.fic_range_stop is None else args.fic_range_stop
    fic_range_step = 1 if args.fic_range_step is None else args.fic_range_step
    fic_range = validate_fic_range(fic_range_start, fic_range_stop, fic_range_step, max_bits=nmpb)

    if layer_precision_info is None and args.bit_width is None:
        raise ValueError("Provide either --bit_width or --layer_precision_info")

    load_model_tuple = (config, args.pretrained_model)

    results: List[Dict[str, Any]] = []
    grid = _build_experiment_grid(aggregation_modes, last_k_values)

    for experiment_config in grid:
        print(
            "[experiment] Running configuration:",
            f"aggregation_mode={experiment_config.aggregation_mode}, last_k={experiment_config.last_k}",
        )
        result = run_single_experiment(
            campaign_module=campaign,
            fk_module=fk,
            load_model_tuple=load_model_tuple,
            x_test_pred_correct=x_test_pred_correct,
            hess_ranking=hess_ranking,
            bit_width=args.bit_width,
            layer_precision_info=layer_precision_info,
            fic_range=fic_range,
            experiment_config=experiment_config,
        )
        results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    json_path, csv_path = save_results(output_dir, timestamp, results)
    heatmap_path = save_heatmap(
        output_dir=output_dir,
        timestamp=timestamp,
        results=results,
        aggregation_modes=aggregation_modes,
        last_k_values=last_k_values,
    )

    print(f"[experiment] Saved JSON results: {json_path}")
    print(f"[experiment] Saved CSV results : {csv_path}")
    if heatmap_path is None:
        print("[experiment] Heat-map not generated.")
    else:
        print(f"[experiment] Saved heat-map   : {heatmap_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensitivity-pointer PrioriFI experiment")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./dense_baseline_fkeras/fkeras_dense_model_58.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to pretrained model weights",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="dense_baseline",
        help="Model identifier string",
    )
    parser.add_argument(
        "--output_dir",
        "--fic_output_dir",
        dest="output_dir",
        type=str,
        default=os.getcwd(),
        help="Directory to save experiment outputs",
    )
    parser.add_argument(
        "--fic_range_start",
        type=int,
        default=None,
        help="Fault injection range start (inclusive)",
    )
    parser.add_argument(
        "--fic_range_stop",
        type=int,
        default=None,
        help="Fault injection range stop (exclusive)",
    )
    parser.add_argument(
        "--fic_range_step",
        type=int,
        default=None,
        help="Fault injection range step",
    )
    parser.add_argument(
        "--bit_width",
        type=int,
        default=None,
        help="Bit width for uniform precision model",
    )
    parser.add_argument(
        "--layer_precision_info",
        type=str,
        default=None,
        help="Mixed-precision layer metadata, e.g. '[(1000,8),(2000,16)]'",
    )
    parser.add_argument(
        "--last_k_values",
        type=str,
        default="1,3,5",
        help="List of positive history sizes, e.g. '1,3,5' or '[1,3,5]'",
    )
    parser.add_argument(
        "--aggregation_modes",
        type=str,
        default="mean,median",
        help="List of aggregation modes (mean, median)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
