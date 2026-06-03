#!/usr/bin/env python3
"""Render small SVG benchmark charts from result JSON artifacts."""

import argparse
import html
import json
from pathlib import Path


ENGINE_COLORS = {
    "vllm": "#2563eb",
    "sglang": "#059669",
    "tensorrt-llm-engine": "#7c3aed",
    "tensorrt-llm": "#a16207",
    "tgi": "#dc2626",
}


FEATURES = [
    ("streaming", "Streaming"),
    ("openai_models", "Models API"),
    ("concurrency", "Concurrency"),
    ("prefix_cache", "Prefix cache"),
    ("json_mode", "JSON mode"),
    ("tool_calling", "Tool calling"),
]


def load_json(path):
    with path.open() as f:
        return json.load(f)


def latest_result_runs(results_dir):
    runs_by_key = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("features_") or path.name.endswith("_smoke.json"):
            continue
        data = load_json(path)
        levels = data.get("levels")
        if not isinstance(levels, list):
            continue
        if data.get("profile") != "homelab":
            continue
        if (data.get("config") or {}).get("workload") != "prefill_heavy":
            continue
        key = (data.get("engine"), data.get("model"))
        if key not in runs_by_key or data.get("timestamp", "") > runs_by_key[key].get("timestamp", ""):
            data["path"] = path
            runs_by_key[key] = data
    return list(runs_by_key.values())


def latest_feature_runs(results_dir):
    features_by_engine = {}
    for path in sorted(results_dir.glob("features_*.json")):
        data = load_json(path)
        if data.get("profile") != "homelab":
            continue
        key = data.get("engine")
        if key not in features_by_engine or data.get("timestamp", "") > features_by_engine[key].get("timestamp", ""):
            data["path"] = path
            features_by_engine[key] = data
    return list(features_by_engine.values())


def best_level(run):
    levels = [level for level in run.get("levels", []) if level.get("ok")]
    return max(levels, key=lambda item: item.get("agg_tokens_per_s") or 0) if levels else None


def metric(run, key):
    level = best_level(run)
    if not level:
        return None
    if key == "tokens_per_joule":
        return (level.get("gpu_metrics") or {}).get("tokens_per_joule")
    return level.get(key)


def label_for(run):
    engine = run.get("engine", "unknown")
    model = run.get("model", "unknown").split("/")[-1]
    if engine == "tensorrt-llm-engine":
        return f"TensorRT-LLM engine ({model})"
    return f"{engine} ({model})"


def svg_bar_chart(title, subtitle, rows, value_key, value_label, out_path):
    width = 960
    left = 250
    top = 96
    row_h = 64
    bar_h = 24
    right = 44
    height = top + row_h * max(1, len(rows)) + 52
    max_value = max([row["value"] for row in rows] or [1])
    scale_w = width - left - right

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#0f172a} .sub{fill:#475569}.axis{fill:#94a3b8}.small{font-size:13px}.label{font-size:15px;font-weight:600}.title{font-size:28px;font-weight:800}.value{font-size:14px;font-weight:700}.note{font-size:12px;fill:#64748b}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="18" y="18" width="924" height="{}" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>'.format(height - 36),
        f'<text x="44" y="54" class="title">{html.escape(title)}</text>',
        f'<text x="44" y="78" class="sub small">{html.escape(subtitle)}</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        value = row["value"]
        bar_w = 0 if max_value == 0 else (value / max_value) * scale_w
        color = ENGINE_COLORS.get(row.get("engine"), "#334155")
        parts.extend(
            [
                f'<text x="44" y="{y + 20}" class="label">{html.escape(row["label"])}</text>',
                f'<text x="44" y="{y + 40}" class="note">{html.escape(row["meta"])}</text>',
                f'<rect x="{left}" y="{y + 5}" width="{scale_w}" height="{bar_h}" rx="4" fill="#e2e8f0"/>',
                f'<rect x="{left}" y="{y + 5}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{color}"/>',
                f'<text x="{left + bar_w + 10 if bar_w < scale_w - 120 else left + bar_w - 112:.1f}" y="{y + 23}" class="value">{value:.2f} {html.escape(value_label)}</text>',
            ]
        )
    parts.append("</svg>\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def feature_mark(value):
    if value is True:
        return ("yes", "#16a34a", "#dcfce7")
    if value is False:
        return ("no", "#dc2626", "#fee2e2")
    return ("manual", "#64748b", "#f1f5f9")


def svg_feature_matrix(reports, out_path):
    width = 1120
    left = 250
    top = 104
    col_w = 130
    row_h = 56
    height = top + row_h * max(1, len(reports)) + 64
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Feature validation matrix">',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#0f172a}.title{font-size:28px;font-weight:800}.sub{font-size:13px;fill:#475569}.head{font-size:12px;font-weight:700;fill:#334155}.engine{font-size:15px;font-weight:700}.model{font-size:12px;fill:#64748b}.cell{font-size:13px;font-weight:800;text-anchor:middle}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="18" y="18" width="1084" height="{}" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>'.format(height - 36),
        '<text x="44" y="54" class="title">Feature validation matrix</text>',
        '<text x="44" y="78" class="sub">Generated from results/features_*.json. "manual" means startup/config log validation is required.</text>',
    ]
    for idx, (_, label) in enumerate(FEATURES):
        x = left + idx * col_w + col_w / 2
        parts.append(f'<text x="{x:.1f}" y="98" class="head" text-anchor="middle">{html.escape(label)}</text>')
    for r_idx, report in enumerate(sorted(reports, key=lambda item: item.get("engine", ""))):
        y = top + r_idx * row_h
        engine = report.get("engine", "unknown")
        model = report.get("model", "unknown").split("/")[-1]
        parts.extend(
            [
                f'<text x="44" y="{y + 21}" class="engine">{html.escape(engine)}</text>',
                f'<text x="44" y="{y + 40}" class="model">{html.escape(model)}</text>',
            ]
        )
        checks = report.get("checks") or {}
        for c_idx, (key, _) in enumerate(FEATURES):
            x = left + c_idx * col_w
            label, fg, bg = feature_mark((checks.get(key) or {}).get("ok"))
            parts.extend(
                [
                    f'<rect x="{x + 22}" y="{y + 7}" width="86" height="30" rx="15" fill="{bg}" stroke="{fg}" stroke-opacity="0.25"/>',
                    f'<text x="{x + 65}" y="{y + 27}" class="cell" fill="{fg}">{label}</text>',
                ]
            )
    parts.append("</svg>\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="docs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = latest_result_runs(results_dir)
    rows = []
    efficiency_rows = []
    for run in runs:
        level = best_level(run)
        if not level:
            continue
        row = {
            "engine": run.get("engine"),
            "label": label_for(run),
            "meta": f"c={level.get('concurrency')} ok={level.get('ok')} model={run.get('model')}",
            "value": float(level.get("agg_tokens_per_s") or 0),
        }
        rows.append(row)
        tpj = metric(run, "tokens_per_joule")
        if tpj is not None:
            efficiency_rows.append({**row, "value": float(tpj)})

    rows.sort(key=lambda item: item["value"], reverse=True)
    efficiency_rows.sort(key=lambda item: item["value"], reverse=True)

    svg_bar_chart(
        "Peak aggregate throughput",
        "Best observed concurrency per refreshed homelab run. Higher is better.",
        rows,
        "agg_tokens_per_s",
        "tok/s",
        out_dir / "bench-throughput.svg",
    )
    svg_bar_chart(
        "Peak serving efficiency",
        "Tokens per joule at each run's best throughput level. Higher is better.",
        efficiency_rows,
        "tokens_per_joule",
        "tok/J",
        out_dir / "bench-efficiency.svg",
    )
    svg_feature_matrix(latest_feature_runs(results_dir), out_dir / "bench-features.svg")

    print(f"[charts] wrote {out_dir / 'bench-throughput.svg'}")
    print(f"[charts] wrote {out_dir / 'bench-efficiency.svg'}")
    print(f"[charts] wrote {out_dir / 'bench-features.svg'}")


if __name__ == "__main__":
    main()
