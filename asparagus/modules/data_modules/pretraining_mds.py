"""Pretraining data module backed directly by MosaicML MDS shards."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

logger = logging.getLogger(__name__)


def _import_streaming():
    try:
        from streaming import StreamingDataset
    except ImportError as exc:  # pragma: no cover - user-facing optional dep error
        raise ImportError(
            "MDSPretrainDataModule requires the optional `mosaicml-streaming` package. "
            "On Nebius, run `bash scripts/install_streaming.sh` after `uv sync --group dcai --extra extras`."
        ) from exc

    try:
        from streaming import Stream
    except ImportError:  # The Nebius install script keeps streaming.__init__ minimal.
        try:
            from streaming.base.stream import Stream
        except ImportError:
            Stream = None

    return StreamingDataset, Stream


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()

    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                logger.warning("Ignoring non-integer %s=%r while resolving MDS world size.", key, value)

    return 1


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    for key in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value:
            try:
                return max(0, int(value))
            except ValueError:
                logger.warning("Ignoring non-integer %s=%r while resolving MDS rank.", key, value)

    return 0


def _nan_clean(data_dict: dict[str, Any]) -> dict[str, Any]:
    img = data_dict["image"]
    if torch.isnan(img).any() or torch.isinf(img).any():
        data_dict["image"] = torch.nan_to_num(img, nan=0.0, posinf=4.0, neginf=-1.0)

    lbl = data_dict.get("label")
    if lbl is not None and (torch.isnan(lbl).any() or torch.isinf(lbl).any()):
        data_dict["label"] = torch.nan_to_num(lbl, nan=0.0, posinf=4.0, neginf=-1.0)

    return data_dict


def _cleanup_streaming_dataset(dataset) -> None:
    iterator = getattr(dataset, "_iterator", None)
    if iterator is not None:
        try:
            iterator.exit()
        except Exception:
            logger.debug("Ignoring StreamingDataset iterator cleanup failure.", exc_info=True)

    executor = getattr(dataset, "_executor", None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.debug("Ignoring StreamingDataset executor cleanup failure.", exc_info=True)

    locals_shm = getattr(dataset, "_locals_shm", None)
    if locals_shm is not None:
        try:
            locals_shm.buf[:4] = np.int32(0).tobytes()
        except Exception:
            logger.debug("Ignoring StreamingDataset shared-memory cleanup failure.", exc_info=True)


def _build_streaming_dataset_class():
    """Create the StreamingDataset subclass lazily so the dependency stays optional."""

    StreamingDataset, _ = _import_streaming()

    class _MDSPretrain(StreamingDataset):
        def __init__(
            self,
            *,
            local: str,
            split: str,
            remote: Optional[str] = None,
            streams: Optional[list[Any]] = None,
            transforms: Optional[Compose] = None,
            return_sample_id: bool = False,
            **streaming_kwargs,
        ):
            if streams is None:
                super().__init__(local=local, remote=remote, split=split, **streaming_kwargs)
            else:
                super().__init__(streams=streams, **streaming_kwargs)

            self.transforms = transforms
            self.return_sample_id = return_sample_id
            self._tag = os.path.join(str(local), str(split))

        def __getitem__(self, sample_id):
            sample_id = int(sample_id)
            if self.return_sample_id:
                return {"sample_id": sample_id}

            sample = super().__getitem__(sample_id)
            return self._build(sample_id, sample)

        def _build(self, sample_id: int, sample: dict[str, Any]) -> dict[str, Any]:
            if "image" not in sample:
                keys = ", ".join(sorted(sample.keys()))
                raise KeyError(f"MDS sample {sample_id} has no 'image' field. Available keys: {keys}")

            image = sample["image"]
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(np.asarray(image))
            if image.dtype != torch.float32:
                image = image.float()
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4:
                raise ValueError(
                    f"Expected MDS image to be 3D or 4D [C,D,H,W]; got shape {tuple(image.shape)} "
                    f"at sample {sample_id}."
                )

            data_dict = {
                "file_path": str(sample.get("file_path", f"{self._tag}/{sample_id:08d}")),
                "image": image.contiguous(),
                "mds_sample_id": sample_id,
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
        mds_remote: Optional[str] = None,
        mds_train_split: str = "train",
        mds_val_split: str = "val",
        mds_streams: Optional[list[dict[str, Any]]] = None,
        num_canonical_nodes: Optional[int] = None,
        predownload: Optional[int] = None,
        prefetch_factor: int = 8,
        cache_limit: Optional[str | int] = None,
        sampling_method: str = "balanced",
        sampling_granularity: int = 1,
        partition_algo: str = "relaxed",
        shuffle_algo: str = "py1e",
        shuffle_seed: int = 9176,
        shuffle_block_size: Optional[int] = None,
        batching_method: str = "random",
        allow_unsafe_types: bool = False,
        assert_disjoint_batches: int = 100,
        train_split=None,
        val_split=None,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        num_samples: Optional[int] = None,
    ):
        super().__init__()
        if mds_root is None:
            raise ValueError(
                "MDSPretrainDataModule requires `mds_root`. Set "
                "`lightning._data_module.mds_root` in the project config."
            )

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.mds_root = str(mds_root)
        self.mds_remote = str(mds_remote) if mds_remote is not None else None
        self.mds_train_split = str(mds_train_split)
        self.mds_val_split = str(mds_val_split)
        self.mds_streams = mds_streams
        self.num_canonical_nodes = num_canonical_nodes
        self.prefetch_factor = int(prefetch_factor)
        self.predownload = predownload
        self.cache_limit = cache_limit
        self.sampling_method = sampling_method
        self.sampling_granularity = int(sampling_granularity)
        self.partition_algo = partition_algo
        self.shuffle_algo = shuffle_algo
        self.shuffle_seed = int(shuffle_seed)
        self.shuffle_block_size = shuffle_block_size
        self.batching_method = batching_method
        self.allow_unsafe_types = bool(allow_unsafe_types)
        self.assert_disjoint_batches = int(assert_disjoint_batches)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_samples = num_samples

        # The standard pretrain pipeline always passes file splits. MDS uses
        # the shard index instead, so these are accepted only for compatibility.
        _ = train_split
        _ = val_split

        logger.info(
            "MDSPretrainDataModule: workers=%s root=%s remote=%s train=%s val=%s",
            self.num_workers,
            self.mds_root,
            self.mds_remote,
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
        self.train_dataset = self._make_dataset(
            Cls,
            split=self.mds_train_split,
            transforms=self.train_transforms,
            shuffle=True,
        )
        self.val_dataset = self._make_dataset(
            Cls,
            split=self.mds_val_split,
            transforms=self.val_transforms,
            shuffle=False,
        )
        self._assert_disjoint_rank_samples(Cls)

    def _resolved_num_canonical_nodes(self) -> int:
        return int(self.num_canonical_nodes) if self.num_canonical_nodes is not None else _distributed_world_size()

    def _resolved_predownload(self) -> int:
        return int(self.predownload) if self.predownload is not None else self.batch_size * self.prefetch_factor

    def _streaming_kwargs(self, shuffle: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "predownload": self._resolved_predownload(),
            "cache_limit": self.cache_limit,
            "sampling_method": self.sampling_method,
            "sampling_granularity": self.sampling_granularity,
            "partition_algo": self.partition_algo,
            "num_canonical_nodes": self._resolved_num_canonical_nodes(),
            "shuffle": shuffle,
            "shuffle_algo": self.shuffle_algo,
            "shuffle_seed": self.shuffle_seed,
            "shuffle_block_size": self.shuffle_block_size,
            "batching_method": self.batching_method,
            "allow_unsafe_types": self.allow_unsafe_types,
        }
        return {key: value for key, value in kwargs.items() if value is not None}

    def _make_streams(self, split: str):
        if not self.mds_streams:
            return None

        _, Stream = _import_streaming()
        if Stream is None:
            raise ImportError("This mosaicml-streaming install does not expose `Stream`; cannot use `mds_streams`.")

        streams = []
        for stream_cfg in self.mds_streams:
            kwargs = dict(stream_cfg)
            kwargs.setdefault("local", self.mds_root)
            kwargs.setdefault("remote", self.mds_remote)
            kwargs.setdefault("split", split)
            streams.append(Stream(**kwargs))
        return streams

    def _make_dataset(
        self,
        Cls,
        *,
        split: str,
        transforms: Optional[Compose],
        shuffle: bool,
        return_sample_id: bool = False,
    ):
        return Cls(
            local=self.mds_root,
            remote=self.mds_remote,
            split=split,
            streams=self._make_streams(split),
            transforms=transforms,
            return_sample_id=return_sample_id,
            **self._streaming_kwargs(shuffle=shuffle),
        )

    def _assert_disjoint_rank_samples(self, Cls) -> None:
        world_size = _distributed_world_size()
        if self.assert_disjoint_batches <= 0 or world_size <= 1:
            return
        if not (dist.is_available() and dist.is_initialized()):
            logger.warning(
                "Skipping MDS disjoint-rank startup assert because torch.distributed is not initialized "
                "(resolved world_size=%s).",
                world_size,
            )
            return

        probe = self._make_dataset(
            Cls,
            split=self.mds_train_split,
            transforms=None,
            shuffle=True,
            return_sample_id=True,
        )
        rank = _distributed_rank()
        sample_limit = self.assert_disjoint_batches * self.batch_size
        ids: list[int] = []
        iterator = iter(probe)
        try:
            for _ in range(sample_limit):
                ids.append(int(next(iterator)["sample_id"]))
        except StopIteration:
            pass
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
            try:
                probe.next_epoch = 0
            except Exception:
                logger.debug("Could not reset StreamingDataset probe epoch.", exc_info=True)
            _cleanup_streaming_dataset(probe)

        gathered: list[list[int]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, ids)

        sets = [set(rank_ids) for rank_ids in gathered]
        for left in range(world_size):
            for right in range(left + 1, world_size):
                overlap = sets[left].intersection(sets[right])
                if overlap:
                    examples = sorted(overlap)[:10]
                    raise RuntimeError(
                        "MDS StreamingDataset rank partitioning is not disjoint: "
                        f"rank {left} and rank {right} share sample ids {examples}. "
                        "Check num_canonical_nodes, batch_size, and distributed launch settings."
                    )

        if rank == 0:
            logger.info(
                "MDS disjoint-rank startup assert passed for %s ranks over %s batches (%s samples/rank).",
                world_size,
                self.assert_disjoint_batches,
                len(ids),
            )

    def _loader_kwargs(self):
        kwargs = {
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def train_dataloader(self):
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

    def teardown(self, stage: Optional[str] = None) -> None:
        for attr in ("train_dataset", "val_dataset"):
            dataset = getattr(self, attr, None)
            if dataset is not None:
                _cleanup_streaming_dataset(dataset)
