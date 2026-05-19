#!/usr/bin/env python3
"""Convert an MDS pretraining split into an Asparagus pretraining task.

This is intentionally a thin format bridge: it reads the MDS ``image`` column,
writes one ``[C, D, H, W]`` torch tensor per sample, and creates the JSON files
that ``asp_pretrain`` already expects. It does not change the SSL objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mds-root", type=Path, required=True, help="Root containing train/val MDS split dirs.")
    parser.add_argument("--output-root", type=Path, required=True, help="ASPARAGUS_DATA root to write into.")
    parser.add_argument("--task-name", default="PT901_FOMO300_MDS", help="Asparagus task folder name to create.")
    parser.add_argument("--train-split", default="train", help="MDS split name for training.")
    parser.add_argument("--val-split", default="val", help="MDS split name for validation.")
    parser.add_argument("--split-name", default="split_mihir_90_10", help="Asparagus split JSON stem.")
    parser.add_argument("--remote", default=None, help="Optional remote MDS root, e.g. s3://bucket/path.")
    parser.add_argument("--batch-size", type=int, default=1, help="StreamingDataset batch size for deterministic iteration.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional max train samples to convert.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional max val samples to convert.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tensors.")
    parser.add_argument("--preserve-dtype", action="store_true", help="Do not cast images to float32 before saving.")
    return parser.parse_args()


def make_dataset(local: Path, split: str, remote: str | None, batch_size: int):
    try:
        from streaming import StreamingDataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'mosaicml-streaming'. Install it in the environment used for conversion."
        ) from exc

    kwargs = {"local": str(local), "split": split, "shuffle": False, "batch_size": batch_size}
    if remote:
        kwargs["remote"] = remote
    return StreamingDataset(**kwargs)


def as_image_tensor(sample: dict, sample_idx: int, preserve_dtype: bool) -> torch.Tensor:
    if "image" not in sample:
        keys = ", ".join(sorted(sample.keys()))
        raise KeyError(f"MDS sample {sample_idx} has no 'image' field. Available keys: {keys}")

    image = torch.as_tensor(sample["image"])
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image shaped [C,D,H,W] or [D,H,W], got {tuple(image.shape)}")

    image = image.contiguous()
    return image if preserve_dtype else image.float()


def convert_split(
    dataset: Iterable[dict],
    split: str,
    task_dir: Path,
    limit: int | None,
    overwrite: bool,
    preserve_dtype: bool,
) -> list[str]:
    split_dir = task_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for idx, sample in enumerate(dataset):
        if limit is not None and idx >= limit:
            break

        out_path = split_dir / f"{idx:08d}.pt"
        if overwrite or not out_path.exists():
            torch.save(as_image_tensor(sample, idx, preserve_dtype), out_path)
        paths.append(str(out_path))

    return paths


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    task_dir = args.output_root / args.task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(args.mds_root, args.train_split, args.remote, args.batch_size)
    val_dataset = make_dataset(args.mds_root, args.val_split, args.remote, args.batch_size)

    train_paths = convert_split(
        train_dataset,
        args.train_split,
        task_dir,
        args.limit_train,
        args.overwrite,
        args.preserve_dtype,
    )
    val_paths = convert_split(
        val_dataset,
        args.val_split,
        task_dir,
        args.limit_val,
        args.overwrite,
        args.preserve_dtype,
    )

    write_json(task_dir / "paths.json", train_paths + val_paths)
    write_json(task_dir / f"{args.split_name}.json", [{"train": train_paths, "val": val_paths}])
    write_json(
        task_dir / "dataset.json",
        {
            "name": args.task_name,
            "metadata": {
                "n_classes": 1,
                "n_modalities": 1,
                "files_target_directory_total": len(train_paths) + len(val_paths),
                "files_train": len(train_paths),
                "files_val": len(val_paths),
            },
            "dataset_config": {
                "task_name": args.task_name,
                "n_classes": 1,
                "n_modalities": 1,
                "source_format": "MDS",
                "source_root": str(args.mds_root),
                "remote": args.remote,
            },
            "preprocessing_config": {
                "conversion": "MDS image column saved as torch tensor",
                "target_shape": "[C,D,H,W]",
            },
            "saving_config": {"save_as_tensor": True},
        },
    )

    print(f"Wrote {len(train_paths)} train and {len(val_paths)} val samples to {task_dir}")
    print(f"Use: task={args.task_name} data.train_split={args.split_name}")


if __name__ == "__main__":
    main()
