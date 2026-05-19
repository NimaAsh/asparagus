#!/usr/bin/env python3
"""Summarize telemetry files emitted by scripts/nebius_pretrain.sbatch."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "[Not Supported]":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p90_idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[p90_idx],
        "max": max(values),
    }


def print_summary(label: str, values: list[float], unit: str = "") -> None:
    stats = summarize(values)
    if stats is None:
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"{label}: mean={stats['mean']:.2f}{suffix}, "
        f"median={stats['median']:.2f}{suffix}, "
        f"p90={stats['p90']:.2f}{suffix}, max={stats['max']:.2f}{suffix}"
    )


def find_column(fieldnames: list[str], prefix: str) -> str | None:
    for field in fieldnames:
        if field.strip().startswith(prefix):
            return field
    return None


def summarize_gpu(path: Path) -> None:
    if not path.exists():
        print(f"GPU CSV missing: {path}")
        return

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    print(f"\nGPU telemetry: {path} ({len(rows)} samples)")
    columns = {
        "GPU util": ("utilization.gpu", "%"),
        "GPU memory util": ("utilization.memory", "%"),
        "GPU memory used": ("memory.used", "MiB"),
        "Power draw": ("power.draw", "W"),
        "SM clock": ("clocks.sm", "MHz"),
        "Memory clock": ("clocks.mem", "MHz"),
        "GPU temperature": ("temperature.gpu", "C"),
    }
    for label, (prefix, unit) in columns.items():
        column = find_column(fields, prefix)
        if column is None:
            continue
        values = [to_float(row.get(column)) for row in rows]
        print_summary(label, [value for value in values if value is not None], unit)


def summarize_vmstat(path: Path) -> None:
    if not path.exists():
        print(f"vmstat log missing: {path}")
        return

    samples: dict[str, list[float]] = {key: [] for key in ["r", "b", "us", "sy", "id", "wa"]}
    fields: list[str] | None = None
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["r", "b"]:
            fields = parts
            continue
        if fields is None or not parts[0].lstrip("-").isdigit():
            continue
        for key in samples:
            if key not in fields:
                continue
            idx = fields.index(key)
            if idx < len(parts):
                value = to_float(parts[idx])
                if value is not None:
                    samples[key].append(value)

    print(f"\nCPU/memory telemetry: {path}")
    print_summary("Run queue", samples["r"])
    print_summary("Blocked tasks", samples["b"])
    print_summary("CPU user", samples["us"], "%")
    print_summary("CPU system", samples["sy"], "%")
    print_summary("CPU idle", samples["id"], "%")
    print_summary("CPU iowait", samples["wa"], "%")


def summarize_iostat(path: Path) -> None:
    if not path.exists():
        print(f"iostat log missing: {path}")
        return

    util_values: list[float] = []
    read_values: list[float] = []
    write_values: list[float] = []
    headers: list[str] | None = None
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Device":
            headers = parts
            continue
        if headers is None or len(parts) != len(headers):
            continue
        row = dict(zip(headers, parts))
        for target, dest in [("%util", util_values), ("rMB/s", read_values), ("wMB/s", write_values)]:
            value = to_float(row.get(target))
            if value is not None:
                dest.append(value)
        # Some sysstat builds use kB/s instead of MB/s.
        if "rMB/s" not in row:
            value = to_float(row.get("rkB/s"))
            if value is not None:
                read_values.append(value / 1024)
        if "wMB/s" not in row:
            value = to_float(row.get("wkB/s"))
            if value is not None:
                write_values.append(value / 1024)

    print(f"\nDisk telemetry: {path}")
    print_summary("Disk util across devices", util_values, "%")
    print_summary("Disk read throughput across devices", read_values, "MB/s")
    print_summary("Disk write throughput across devices", write_values, "MB/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_ids", nargs="+", help="Slurm job IDs to summarize")
    parser.add_argument("--slurms-dir", default="slurms", type=Path)
    args = parser.parse_args()

    for job_id in args.job_ids:
        prefix = args.slurms_dir / job_id
        print(f"\n=== Job {job_id} ===")
        summarize_gpu(prefix.with_name(f"{prefix.name}_gpu.csv"))
        summarize_vmstat(prefix.with_name(f"{prefix.name}_vmstat.log"))
        summarize_iostat(prefix.with_name(f"{prefix.name}_iostat.log"))


if __name__ == "__main__":
    main()
