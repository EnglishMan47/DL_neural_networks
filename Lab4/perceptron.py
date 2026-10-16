"""Двухслойный ReLU-перцептрон и градиентный спуск"""

from dataclasses import dataclass

import numpy as np


def initialize(n_features, hidden, seed=2026):
    """Фиксированные начальные веса из интервала [-0.1, 0.1]"""
    if not all(
        isinstance(v, (int, np.integer)) and v > 0 for v in (n_features, hidden)
    ):
        raise ValueError("Размеры слоёв должны быть положительными целыми числами")
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(-0.1, 0.1, (n_features, hidden)),
        np.zeros(hidden),
        rng.uniform(-0.1, 0.1, hidden),
        np.zeros(1),
    ]


def predict(x, params):
    w1, b1, w2, b2 = params
    return np.maximum(np.asarray(x, dtype=float) @ w1 + b1, 0) @ w2 + b2[0]


def loss_gradient(x, y, params):
    """J = sum((prediction - y)**2). В нуле H(0) = 1"""
    w1, b1, w2, b2 = params
    z = x @ w1 + b1
    hidden = np.maximum(z, 0)
    error = hidden @ w2 + b2[0] - y
    output_gradient = 2 * error
    hidden_gradient = output_gradient[:, None] * w2 * (z >= 0)
    gradients = [
        x.T @ hidden_gradient,
        hidden_gradient.sum(axis=0),
        hidden.T @ output_gradient,
        np.array([output_gradient.sum()]),
    ]
    return float(error @ error), gradients


def gradient_norm(gradients):
    return float(max(np.max(abs(g)) for g in gradients))


@dataclass
class FitResult:
    params: list
    iterations: int
    converged: bool
    loss: float
    grad_norm: float
    history: np.ndarray


def train(
    x,
    y,
    hidden,
    *,
    learning_rate=1e-6,
    tolerance=1e-3,
    max_iter=30000,
    schedule="constant",
    decay=1000.0,
    seed=2026,
    initial=None,
):
    """Остановка при ||gradient J||_inf < tolerance или достижении лимита."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or not len(x) or x.shape[1] == 0 or y.shape != (len(x),):
        raise ValueError("Несогласованные X и y")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("X и y должны быть конечными")
    if (
        not np.isfinite([learning_rate, tolerance, decay]).all()
        or learning_rate <= 0
        or tolerance <= 0
        or decay <= 0
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 0
        or not isinstance(hidden, (int, np.integer))
        or hidden <= 0
        or schedule not in ("constant", "decay")
    ):
        raise ValueError("Неверные параметры обучения")
    params = (
        initialize(x.shape[1], hidden, seed)
        if initial is None
        else [np.array(p, float, copy=True) for p in initial]
    )
    expected = [(x.shape[1], hidden), (hidden,), (hidden,), (1,)]
    if [p.shape for p in params] != expected or not all(
        np.isfinite(p).all() for p in params
    ):
        raise ValueError("Неверная форма или значения начальных весов")
    history = []
    for step in range(max_iter + 1):
        loss, gradients = loss_gradient(x, y, params)
        norm = gradient_norm(gradients)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
            raise FloatingPointError(
                "Градиентный спуск расходится: требуется меньший шаг"
            )
        lr = (
            learning_rate
            if schedule == "constant"
            else learning_rate / (1 + step / decay)
        )
        converged = norm < tolerance
        if step % 100 == 0 or converged or step == max_iter:
            history.append([step, loss, norm, lr])
        if converged or step == max_iter:
            return FitResult(params, step, converged, loss, norm, np.array(history))
        for parameter, gradient in zip(params, gradients):
            parameter -= lr * gradient


def save_model(path, result):
    """Сохранение весов для исходных, непреобразованных признаков"""
    np.savez_compressed(
        path,
        w1=result.params[0],
        b1=result.params[1],
        w2=result.params[2],
        b2=result.params[3],
    )


def predict_saved(path, x):
    with np.load(path, allow_pickle=False) as data:
        params = [data[key] for key in ("w1", "b1", "w2", "b2")]
        return predict(x, params)
