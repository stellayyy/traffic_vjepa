import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TaxiBJDataset, load_array, to_tchw
from .metrics import evaluation_metrics
from .models import TaxiVJEPA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--key", default="data")
    parser.add_argument("--layout", default="TCHW")
    parser.add_argument("--out", default="checkpoints/bj13_vjepa_vit")
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--start-lr", type=float, default=1e-6)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--pred-depth", type=int, default=4)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--predictor-dim", type=int, default=192)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--tubelet-size", type=int, default=1)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--teacher-momentum-end", type=float, default=1.0)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--flow-loss", choices=("l1", "mse"), default="l1")
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="taxi-vjepa-vit")
    parser.add_argument("--wandb-run", default=None)
    return parser.parse_args()


def build_model(args, channels, grid):
    return TaxiVJEPA(
        channels, args.embed_dim, args.enc_depth, args.pred_depth, args.heads,
        args.patch, args.context, args.horizon, grid, args.predictor_dim,
        args.tubelet_size, args.flow_depth,
    )


def denormalize(values, dataset):
    mean = dataset.mean.numpy()[:, :, None, :, :]
    std = dataset.std.numpy()[:, :, None, :, :]
    return values * std + mean


def scheduled_lr(step, total_steps, warmup_steps, start_lr, peak_lr, final_lr):
    """Linear warmup followed by cosine decay, evaluated once per optimizer step."""
    if warmup_steps > 0 and step < warmup_steps:
        progress = step / max(warmup_steps - 1, 1)
        return start_lr + progress * (peak_lr - start_lr)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return final_lr + 0.5 * (peak_lr - final_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, loader, device, train_set, args):
    model.eval()
    predictions, targets = [], []
    normalized_squared_error = 0.0
    elements = 0
    for batch in loader:
        future = batch["future"].to(device)
        output = model(batch["context"].to(device), mask_ratio=args.mask_ratio)["flow"]
        normalized_squared_error += F.mse_loss(output, future, reduction="sum").item()
        elements += future.numel()
        predictions.append(output.cpu().numpy())
        targets.append(batch["future"].numpy())
    pred = denormalize(np.concatenate(predictions), train_set)
    true = denormalize(np.concatenate(targets), train_set)
    metrics = evaluation_metrics(
        pred, true, (train_set.raw_min, train_set.raw_max), args.null_val
    )
    metrics["selection_mse"] = normalized_squared_error / elements
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    array = to_tchw(load_array(args.data, args.key), args.layout)
    train_set = TaxiBJDataset(array, args.context, args.horizon, "train")
    val_set = TaxiBJDataset(
        array, args.context, args.horizon, "val", mean=train_set.mean, std=train_set.std
    )
    np.savez(
        output_dir / "norm.npz", mean=train_set.mean.numpy(), std=train_set.std.numpy(),
        train_min=np.array(train_set.raw_min), train_max=np.array(train_set.raw_max),
    )
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers)
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Train/validation split is too short for context+horizon")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = build_model(args, array.shape[1], array.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_epochs * len(train_loader), total_steps)
    global_step = 0
    best = float("inf")
    history = []
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run,
            config=vars(args), dir=str(output_dir),
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"jepa": 0.0, "flow": 0.0, "total": 0.0}
        for batch in train_loader:
            current_lr = scheduled_lr(
                global_step, total_steps, warmup_steps,
                args.start_lr, args.lr, args.final_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            result = model(
                batch["context"].to(device), batch["future"].to(device),
                args.mask_ratio, args.flow_weight, args.flow_loss,
            )
            optimizer.zero_grad(set_to_none=True)
            result["losses"]["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress = global_step / max(total_steps - 1, 1)
            momentum = args.teacher_momentum_end - (
                args.teacher_momentum_end - args.teacher_momentum
            ) * (math.cos(math.pi * progress) + 1.0) / 2.0
            model.jepa.update_teacher(momentum)
            global_step += 1
            for name in sums:
                sums[name] += float(result["losses"][name].detach())

        val = validate(model, val_loader, device, train_set, args)
        record = {"epoch": epoch}
        record["train/lr"] = current_lr
        record.update({f"train/{k}": v / len(train_loader) for k, v in sums.items()})
        record.update({f"val/{k}": v for k, v in val.items()})
        history.append(record)
        print(record)
        if wandb_run:
            wandb_run.log(record, step=epoch)
        checkpoint = {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                      "best_selection_mse": min(best, val["selection_mse"])}
        torch.save(checkpoint, output_dir / "latest.pt")
        # Both L1 and MSE runs use the same normalized validation MSE criterion.
        if val["selection_mse"] < best:
            best = val["selection_mse"]
            torch.save(checkpoint, output_dir / "best.pt")

        (output_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(item) for item in history), encoding="utf-8"
        )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
