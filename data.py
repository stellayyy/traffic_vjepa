from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_array(path, key=None):
    path = Path(path)
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            keys = [key] if key else []
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset): keys.append(name)
            f.visititems(visit)
            if not keys: raise ValueError("No dataset found in HDF5 file")
            return np.asarray(f[keys[0]])
    return np.load(path)

# 接受4D和5D的数组，并返回[T, C, H, W]
def to_tchw(a, layout=None):
    """Convert traffic arrays to canonical [T, C, H, W] format."""
    a = np.asarray(a)

    if a.ndim == 4:
        if layout is None:
            if a.shape[1] <= 8:
                layout = "TCHW"
            elif a.shape[-1] <= 8:
                layout = "THWC"
            else:
                raise ValueError(
                    f"Cannot infer channel axis from shape {a.shape}; "
                    "please specify --layout TCHW or --layout THWC"
                )

        layout = layout.upper()

        if set(layout) != set("TCHW") or len(layout) != 4:
            raise ValueError(
                f"4D layout must be a permutation of TCHW, got {layout}"
            )

        axes = {name: index for index, name in enumerate(layout)}

        return np.transpose(
            a,
            [
                axes["T"],
                axes["C"],
                axes["H"],
                axes["W"],
            ],
        )

    if a.ndim == 5:
        if layout is None:
            if a.shape[-1] <= 8:
                layout = "NTHWC"
            elif a.shape[2] <= 8:
                layout = "NTCHW"
            elif a.shape[1] <= 8:
                layout = "TNCHW"
            else:
                raise ValueError(
                    f"Cannot infer layout from 5D shape {a.shape}"
                )

        layout = layout.upper()

        if set(layout) != set("NTHWC") or len(layout) != 5:
            raise ValueError(
                f"5D layout must be a permutation of NTHWC, got {layout}"
            )

        axes = {name: index for index, name in enumerate(layout)}

        if a.shape[axes["N"]] != 1:
            raise ValueError(
                "Conversion of a 5D array currently requires N=1, "
                f"but got shape {a.shape}"
            )

        a = np.transpose(
            a,
            [
                axes["N"],
                axes["T"],
                axes["C"],
                axes["H"],
                axes["W"],
            ],
        )

        return a[0]

    raise ValueError(
        f"Expected a 4D or 5D array, got shape {a.shape}"
    )


class TaxiBJDataset(Dataset):
    def __init__(self, array, context=12, horizon=3, split="train", train_end=None, val_end=None, mean=None, std=None):
        self.data = torch.as_tensor(array, dtype=torch.float32)
        n = self.data.shape[0]
        train_end = train_end or int(n * .7); val_end = val_end or int(n * .8)
        bounds = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}
        lo, hi = bounds[split]
        self.data = self.data[lo:hi]
        self.raw_min = float(self.data.min())
        self.raw_max = float(self.data.max())
        self.context, self.horizon = context, horizon
        self.mean = mean if mean is not None else self.data.mean((0, 2, 3), keepdim=True)
        self.std = std if std is not None else self.data.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
        self.data = (self.data - self.mean) / self.std
    def __len__(self): return max(0, self.data.shape[0] - self.context - self.horizon + 1)
    def __getitem__(self, i):
        x = self.data[i:i+self.context].permute(1, 0, 2, 3)
        y = self.data[i+self.context:i+self.context+self.horizon].permute(1, 0, 2, 3)
        return {"context": x, "future": y}
