import torch

from taxi_vjepa_vit.models import TaxiVJEPA


def main():
    torch.manual_seed(0)
    model = TaxiVJEPA(
        in_channels=2, embed_dim=48, enc_depth=2, pred_depth=2, heads=3,
        patch=4, frames=4, horizon=2, grid=8, hidden=48,
        tubelet_size=1, flow_depth=2,
    )
    context = torch.randn(2, 2, 4, 8, 8)
    future = torch.randn(2, 2, 2, 8, 8)
    for loss_name in ("l1", "mse"):
        model.zero_grad(set_to_none=True)
        output = model(context, future, flow_loss=loss_name)
        assert output["flow"].shape == future.shape
        assert output["pred_tokens"].shape == output["target_tokens"].shape
        output["losses"]["total"].backward()
        assert any(p.grad is not None for p in model.jepa.student.parameters())
        assert all(p.grad is None for p in model.jepa.teacher.parameters())
    before = next(model.jepa.teacher.parameters()).clone()
    with torch.no_grad():
        next(model.jepa.student.parameters()).add_(0.01)
    model.jepa.update_teacher(0.9)
    assert not torch.equal(before, next(model.jepa.teacher.parameters()))
    print("TaxiVJEPA smoke test passed")


if __name__ == "__main__":
    main()
