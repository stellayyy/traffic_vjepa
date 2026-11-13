import numpy as np


def regression_metrics(pred, true, null_val=0.0):
    """Match reDerf evaluation: masked physical-scale metrics and ratio MAPE."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(true, dtype=np.float64)
    mask = np.ones_like(y, dtype=bool) if np.isnan(null_val) else y != null_val
    if not np.any(mask):
        raise ValueError("No valid targets remain after applying null_val")
    diff = p[mask] - y[mask]
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
        "mape": float(np.mean(np.abs(diff) / np.maximum(np.abs(y[mask]), 1e-5))),
    }


def map_to_01(values, train_minmax):
    values = np.asarray(values)
    lo, hi = train_minmax
    denominator = hi - lo
    if abs(denominator) < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / denominator, 0.0, 1.0)


def simple_ssim(pred, true, data_range=1.0):
    """Exact global-per-sample SSIM used by reDerf's GLA runners."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    axes = tuple(range(1, pred.ndim))
    mu_x = pred.mean(axis=axes)
    mu_y = true.mean(axis=axes)
    expand = (-1, *([1] * (pred.ndim - 1)))
    var_x = pred.var(axis=axes)
    var_y = true.var(axis=axes)
    covariance = ((pred - mu_x.reshape(expand)) * (true - mu_y.reshape(expand))).mean(axis=axes)
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )
    return float(np.mean(score))


def evaluation_metrics(pred, true, train_minmax, null_val=0.0):
    result = regression_metrics(pred, true, null_val=null_val)
    score = simple_ssim(map_to_01(pred, train_minmax), map_to_01(true, train_minmax))
    result.update({"ssim": score, "ssim_01": score})
    return result
