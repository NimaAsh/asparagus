import sys
import types

import numpy as np


class FakeStreamingDataset:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.next_epoch = 0
        FakeStreamingDataset.instances.append(self)

    def __getitem__(self, sample_id):
        return {"image": np.zeros((2, 3, 4), dtype=np.float32), "source_id": int(sample_id)}

    def __iter__(self):
        for sample_id in (5, 42):
            yield self.__getitem__(sample_id)


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def install_fake_streaming(monkeypatch):
    module = types.ModuleType("streaming")
    module.StreamingDataset = FakeStreamingDataset
    module.Stream = FakeStream
    monkeypatch.setitem(sys.modules, "streaming", module)
    FakeStreamingDataset.instances.clear()


def test_mds_dataset_passes_streaming_partition_knobs(monkeypatch):
    install_fake_streaming(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("SLURM_NTASKS", raising=False)

    from asparagus.modules.data_modules.pretraining_mds import MDSPretrainDataModule

    dm = MDSPretrainDataModule(
        batch_size=4,
        num_workers=0,
        mds_root="/mds",
        train_transforms=None,
        val_transforms=None,
        assert_disjoint_batches=0,
    )
    dm.setup("fit")

    train_kwargs = dm.train_dataset.kwargs
    assert train_kwargs["local"] == "/mds"
    assert train_kwargs["split"] == "train"
    assert train_kwargs["batch_size"] == 4
    assert train_kwargs["predownload"] == 32
    assert train_kwargs["num_canonical_nodes"] == 8
    assert train_kwargs["shuffle"] is True

    val_kwargs = dm.val_dataset.kwargs
    assert val_kwargs["split"] == "val"
    assert val_kwargs["shuffle"] is False


def test_mds_dataset_preserves_global_sample_id_and_transforms_once(monkeypatch):
    install_fake_streaming(monkeypatch)
    calls = []

    def transform(data):
        calls.append(data["mds_sample_id"])
        data["transforms_applied"]["fake"] = 1
        return data

    from asparagus.modules.data_modules.pretraining_mds import MDSPretrainDataModule

    dm = MDSPretrainDataModule(
        batch_size=2,
        num_workers=0,
        mds_root="/mds",
        train_transforms=transform,
        val_transforms=None,
        num_canonical_nodes=1,
        assert_disjoint_batches=0,
    )
    dm.setup("fit")

    row = next(iter(dm.train_dataset))

    assert row["file_path"] == "/mds/train/00000005"
    assert row["mds_sample_id"] == 5
    assert row["image"].shape == (1, 2, 3, 4)
    assert row["transforms_applied"] == {"fake": 1}
    assert calls == [5]
