import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TaxiBJDataset, load_array, to_tchw
from .metrics import evaluation_metrics
from .models import TaxiVJEPA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--key", default="data")
    parser.add_argument("--layout", default="TCHW")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--null-val", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    array = to_tchw(load_array(args.data, args.key), args.layout)
    train_set = TaxiBJDataset(array, cfg["context"], cfg["horizon"], "train")
    test_set = TaxiBJDataset(
        array, cfg["context"], cfg["horizon"], "test",
        mean=train_set.mean, std=train_set.std,
    )
    model = TaxiVJEPA(
        array.shape[1], cfg["embed_dim"], cfg["enc_depth"], cfg["pred_depth"],
        cfg["heads"], cfg["patch"], cfg["context"], cfg["horizon"],
        array.shape[-1], cfg.get("predictor_dim", cfg["embed_dim"]),
        cfg.get("tubelet_size", 1), cfg.get("flow_depth", 4),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in DataLoader(test_set, args.batch_size):
            output = model(batch["context"].to(args.device), mask_ratio=cfg["mask_ratio"])
            predictions.append(output["flow"].cpu().numpy())
            targets.append(batch["future"].numpy())
    mean = train_set.mean.numpy()[:, :, None, :, :]
    std = train_set.std.numpy()[:, :, None, :, :]
    pred = np.concatenate(predictions) * std + mean
    true = np.concatenate(targets) * std + mean
    null_val = cfg.get("null_val", 0.0) if args.null_val is None else args.null_val
    print(evaluation_metrics(pred, true, (train_set.raw_min, train_set.raw_max), null_val))


if __name__ == "__main__":
    main()
