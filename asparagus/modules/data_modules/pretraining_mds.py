"""Pretraining data module backed by mosaicml-streaming MDS shards.

Drop-in replacement for ``PretrainDataModule`` that reads samples directly
from an MDS shard directory instead of from one ``.pt`` file per sample. The
``mds_to_asparagus_pretrain.py`` converter produced 34k individual ``.pt``
files for FOMO300 which broke the locality MDS shards were designed to give
us: each ``__getitem__`` becomes its own file open and pickle, the page
cache cannot keep up once Torch_Spatial is moved to the GPU, and disk reads
become the binding constraint.

This module bypasses the converter entirely and streams from the original
shards. Each DataLoader worker iterates a private subset of shards via
``mosaicml-streaming`` (``streaming.StreamingDataset``), which is an
``IterableDataset``. Random ordering comes from ``shuffle=True`` on the
underlying StreamingDataset, so we deliberately do not wrap it in a
``RandomSampler``. That mirrors the ``num_samples=999999, replacement=True``
infinite-stream behavior the original module relied on while letting the
shard reader do sequential I/O.

Behavior matches ``PretrainDataModule``:

* Emits dicts of ``{"file_path", "image", "label" (optional after transforms),
  "transforms_applied"}``.
* Images are returned as float32 ``[C, D, H, W]`` torch tensors. 3D arrays are
  auto-unsqueezed to add a channel dim.
* NaN/Inf values in image (and label, if present after transforms) are
  replaced via ``nan_to_num``.
* Accepts and silently ignores the legacy ``train_split`` / ``val_split``
  positional kwargs the asparagus pipeline passes from
  ``prepare_standard_experiment``; the shard layout is configured via
  ``mds_root`` / ``mds_train_split`` / ``mds_val_split`` instead.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


def _import_streaming_dataset():
    try:
        from streaming import StreamingDataset
    except ImportError as exc:  # pragma: no cover - clearer error for users
        raise ImportError(
            "MDSPretrainDataModule requires the optional 'mosaicml-streaming' "
            "package. Install it with `uv add mosaicml-streaming` or include "
            "it in your environment."
        ) from exc
    return StreamingDataset


def _nan_clean(data_dict):
    img = data_dict["image"]
    if torch.isnan(img).any() or torch.isinf(img).any():
        data_dict["image"] = torch.nan_to_num(img, nan=0.0, posinf=4.0, neginf=-1.0)
    lbl = data_dict.get("label")
    if lbl is not None and (torch.isnan(lbl).any() or torch.isinf(lbl).any()):
        data_dict["label"] = torch.nan_to_num(lbl, nan=0.0, posinf=4.0, neginf=-1.0)
    return data_dict


def _build_streaming_dataset_class():
    """Create the StreamingDataset subclass lazily.

    The base class lives in the optional ``streaming`` package; defining the
    subclass at module import time would force everyone importing
    ``asparagus.modules.data_modules`` to have ``mosaicml-streaming`` installed
    just to access ``PretrainDataModule`` (the legacy module).
    """

    StreamingDataset = _import_streaming_dataset()

    class _MDSPretrain(StreamingDataset):
        def __init__(
            self,
            local: str,
            split: str,
            transforms: Optional[Compose] = None,
            shuffle: bool = False,
            batch_size: int = 1,
        ):
            super().__init__(
                local=local,
                split=split,
                shuffle=shuffle,
                batch_size=batch_size,
            )
            self.transforms = transforms
            self._tag = f"{local}/{split}"

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            return self._build(idx, sample)

        def __iter__(self):
            for idx, sample in enumerate(super().__iter__()):
                yield self._build(idx, sample)

        def _build(self, idx, sample):
            image = sample["image"]
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(np.asarray(image, dtype=np.float32))
            if image.dtype != torch.float32:
                image = image.float()
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4:
                raise ValueError(
                    f"Expected MDS image to be 3D or 4D [C,D,H,W]; got shape "
                    f"{tuple(image.shape)} at index {idx}"
                )
            image = image.contiguous()

            data_dict = {
                "file_path": f"{self._tag}/{idx:08d}",
                "image": image,
                "transforms_applied": {},
            }
            if self.transforms is not None:
                data_dict = self.transforms(data_dict)
            return _nan_clean(data_dict)

    return _MDSPretrain


class MDSPretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        mds_root: Optional[str] = None,
        mds_train_split: str = "train",
        mds_val_split: str = "val",
        train_split=None,
        val_split=None,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        num_samples: Optional[int] = None,
    ):
        super().__init__()
        if mds_root is None:
            raise ValueError(
                "MDSPretrainDataModule requires `mds_root`. Set it via the "
                "project config's `lightning._data_module.mds_root` (env "
                "interpolation OK)."
            )
        self.mds_root = str(mds_root)
        self.mds_train_split = str(mds_train_split)
        self.mds_val_split = str(mds_val_split)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_samples = num_samples

        # Legacy positional kwargs from prepare_standard_experiment. We do not
        # need them because the shard root drives sample enumeration, but the
        # asparagus pretrain pipeline always passes them as keyword args.
        _ = train_split
        _ = val_split

        logging.info(
            "MDSPretrainDataModule: %s workers; root=%s train=%s val=%s",
            self.num_workers,
            self.mds_root,
            self.mds_train_split,
            self.mds_val_split,
        )

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
            return
        if stage == "test":
            raise NotImplementedError("Test stage not supported for MDSPretrainDataModule.")
        if stage == "predict":
            raise NotImplementedError("Predict stage not supported for MDSPretrainDataModule.")

    def setup_fit(self):
        Cls = _build_streaming_dataset_class()
        self.train_dataset = Cls(
            local=self.mds_root,
            split=self.mds_train_split,
            transforms=self.train_transforms,
            shuffle=True,
            batch_size=self.batch_size,
        )
        self.val_dataset = Cls(
            local=self.mds_root,
            split=self.mds_val_split,
            transforms=self.val_transforms,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def _loader_kwargs(self):
        kwargs = {
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 8
        return kwargs

    def train_dataloader(self):
        # StreamingDataset is an IterableDataset and handles its own random
        # ordering + DDP sharding. We deliberately do NOT pass a sampler.
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            **self._loader_kwargs(),
        )
