#!/usr/bin/env python3
"""Generate benchmark summary markdown from result JSON artifacts."""

import argparse
import json
from pathlib import Path


ENGINE_IMAGE_KEYS = {
    "sglang": "sglang",
    "tensorrt-llm": "tensorrt_llm",
    "tensorrt-llm-engine": "tensorrt_llm",
    "tgi": "tgi",
    "vllm": "vllm",
}

FEATURE_LABELS = {
    "streaming": "Streaming",
    "openai_models": "OpenAI Models API",
    "concurrency": "Concurrent Requests",
    "prefix_cache": "Prefix Cache Smoke",
    "json_mode": "JSON Mode",
    "tool_calling": "Tool Calling",
    "speculative": "Speculative Decoding",
}


def load_json(path):
    with path.open() as f:
        return json.load(f)


def fmt(value, digits=1):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bool_mark(value):
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "manual"
    return str(value)


def image_for(run):
    engine = run.get("engine", "")
    image_key = ENGINE_IMAGE_KEYS.get(engine, engine)
    return (run.get("images") or {}).get(image_key) or ""


def result_files(results_dir):
    return sorted(
        path
        for path in results_dir.glob("*.json")
        if not path.name.startswith("features_") and path.name != "image-digests.json"
    )


def feature_files(results_dir):
    return sorted(results_dir.glob("features_*.json"))


def load_runs(results_dir):
    runs = []
    for path in result_files(results_dir):
        try:
            data = load_json(path)
        except Exception as e:
            runs.append({"path": path, "load_error": str(e)})
            continue
        if not isinstance(data.get("levels"), list):
            continue
        data["path"] = path
        runs.append(data)
    return runs


def load_features(results_dir):
    reports = []
    for path in feature_files(results_dir):
        try:
            data = load_json(path)
        except Exception as e:
            reports.append({"path": path, "load_error": str(e)})
            continue
        data["path"] = path
        reports.append(data)
    return reports


def best_level(run):
    levels = [level for level in run.get("levels", []) if level.get("ok")]
    if not levels:
        return None
    return max(levels, key=lambda item: item.get("agg_tokens_per_s") or 0)


def gpu_metric(level, name, mode="avg"):
    metrics = level.get("gpu_metrics") or {}
    value = metrics.get(name)
    if not isinstance(value, dict):
        return None
    return value.get(mode)


def render_summary(runs):
    lines = [
        "# Benchmark Summary",
        "",
        "Generated from `results/*.json`.",
        "",
    ]
    if not runs:
        lines.extend(["No benchmark result JSON files were found.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "## Runs",
            "",
            "| Result | Engine | Profile | Model | Timestamp | Image | Workload | Prompt Mode |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for run in sorted(runs, key=lambda item: item.get("timestamp", "")):
        config = run.get("config") or {}
        lines.append(
            "| "
            f"`{run['path']}` | `{run.get('engine', '')}` | `{run.get('profile', '')}` | "
            f"`{run.get('model', '')}` | `{run.get('timestamp', '')}` | `{image_for(run)}` | "
            f"`{config.get('workload', 'legacy')}` | `{config.get('prompt_mode', 'repeated')}` |"
        )

    lines.extend(
        [
            "",
            "## Best Throughput By Run",
            "",
            "| Engine | Model | Profile | Concurrency | OK | Failed | Agg tok/s | TTFT p95 (ms) | E2E p95 (s) | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in sorted(runs, key=lambda item: (item.get("engine", ""), item.get("model", ""))):
        level = best_level(run)
        if level is None:
            continue
        vram_mib = gpu_metric(level, "vram_used_mib", "max")
        vram_gib = vram_mib / 1024 if vram_mib is not None else None
        metrics = level.get("gpu_metrics") or {}
        lines.append(
            "| "
            f"`{run.get('engine', '')}` | `{run.get('model', '')}` | `{run.get('profile', '')}` | "
            f"{fmt(level.get('concurrency'), 0)} | {fmt(level.get('ok'), 0)} | {fmt(level.get('failed'), 0)} | "
            f"{fmt(level.get('agg_tokens_per_s'), 1)} | {fmt(level.get('ttft_ms_p95'), 1)} | "
            f"{fmt(level.get('e2e_s_p95'), 3)} | {fmt(gpu_metric(level, 'gpu_util_percent'), 1)} | "
            f"{fmt(vram_gib, 2)} | {fmt(gpu_metric(level, 'power_watts'), 1)} | "
            f"{fmt(metrics.get('tokens_per_joule'), 4)} |"
        )

    lines.extend(["", "## Per-Concurrency Results", ""])
    for run in sorted(runs, key=lambda item: (item.get("engine", ""), item.get("timestamp", ""))):
        lines.extend(
            [
                f"### {run.get('engine', 'unknown')} / {run.get('model', 'unknown')} / {run.get('profile', 'unknown')}",
                "",
                f"Source: `{run['path']}`",
                "",
                "| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Req/s | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for level in run.get("levels", []):
            vram_mib = gpu_metric(level, "vram_used_mib", "max")
            vram_gib = vram_mib / 1024 if vram_mib is not None else None
            metrics = level.get("gpu_metrics") or {}
            lines.append(
                "| "
                f"{fmt(level.get('concurrency'), 0)} | {fmt(level.get('ok'), 0)} | {fmt(level.get('failed'), 0)} | "
                f"{fmt(level.get('ttft_ms_p50'), 1)} | {fmt(level.get('ttft_ms_p95'), 1)} | "
                f"{fmt(level.get('e2e_s_p50'), 3)} | {fmt(level.get('e2e_s_p95'), 3)} | "
                f"{fmt(level.get('mean_tps_per_req'), 1)} | {fmt(level.get('agg_tokens_per_s'), 1)} | "
                f"{fmt(level.get('throughput_req_s'), 2)} | {fmt(gpu_metric(level, 'gpu_util_percent'), 1)} | "
                f"{fmt(vram_gib, 2)} | {fmt(gpu_metric(level, 'power_watts'), 1)} | "
                f"{fmt(metrics.get('tokens_per_joule'), 4)} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_features(reports):
    lines = [
        "# Feature Matrix",
        "",
        "Generated from `results/features_*.json`.",
        "",
    ]
    if not reports:
        lines.extend(
            [
                "No feature validation artifacts were found.",
                "",
                "Run `just features engine=<engine>` after starting an engine to populate this matrix.",
                "",
            ]
        )
        return "\n".join(lines)

    headers = ["Engine", "Model", "Profile", *FEATURE_LABELS.values()]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for report in sorted(reports, key=lambda item: (item.get("engine", ""), item.get("timestamp", ""))):
        checks = report.get("checks") or {}
        cells = [
            f"`{report.get('engine', '')}`",
            f"`{report.get('model', '')}`",
            f"`{report.get('profile', '')}`",
        ]
        for key in FEATURE_LABELS:
            cells.append(bool_mark((checks.get(key) or {}).get("ok")))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Details", ""])
    for report in sorted(reports, key=lambda item: (item.get("engine", ""), item.get("timestamp", ""))):
        checks = report.get("checks") or {}
        lines.extend(
            [
                f"### {report.get('engine', 'unknown')} / {report.get('model', 'unknown')} / {report.get('profile', 'unknown')}",
                "",
                f"Source: `{report['path']}`",
                "",
                "| Feature | Supported | Detail |",
                "|---|---|---|",
            ]
        )
        for key, label in FEATURE_LABELS.items():
            check = checks.get(key) or {}
            detail = str(check.get("detail") or "").replace("\n", " ")
            lines.append(f"| {label} | {bool_mark(check.get('ok'))} | {detail} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--benchmark-dir", default="benchmark")
    parser.add_argument("--summary-out", default="summary.md")
    parser.add_argument("--features-out", default="features.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    benchmark_dir = Path(args.benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    features = load_features(results_dir)

    summary_path = benchmark_dir / args.summary_out
    features_path = benchmark_dir / args.features_out
    summary_path.write_text(render_summary(runs), encoding="utf-8")
    features_path.write_text(render_features(features), encoding="utf-8")

    print(f"[summary] wrote {summary_path}")
    print(f"[summary] wrote {features_path}")


if __name__ == "__main__":
    main()
