"""Золотое сечение и обучение трёх моделей с численными производными"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PARAMETERS = {
    "linear": ("w1", "b1"),
    "quadratic": ("w1", "w2", "b1"),
    "relu": ("w11", "b11", "w12", "b12", "w21", "w22", "b21"),
}
WEIGHT_MASK = {
    "linear": np.array([1.0, 0.0]),
    "quadratic": np.array([1.0, 1.0, 0.0]),
    "relu": np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
}


def target_function(z):
    z = np.asarray(z, dtype=float)
    return (
        1 - 0.5 * np.exp(-10 * (z - 0.65) ** 2) - 0.45 * np.exp(-50 * (z - 0.35) ** 2)
    )


@dataclass
class Minimum:
    x: float
    fun: float
    left: float
    right: float
    iterations: int
    evaluations: int


def golden_section(function, left, right, *, tolerance=1e-10, max_iter=1000):
    """Минимум на унимодальном отрезке; длина финального отрезка <= tolerance"""
    if (
        not np.isfinite([left, right, tolerance]).all()
        or left >= right
        or tolerance <= 0
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("Неверный отрезок, точность или лимит")
    ratio = (np.sqrt(5) - 1) / 2
    a, b = float(left), float(right)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = float(function(c)), float(function(d))
    evaluations = 2
    for step in range(max_iter + 1):
        if not np.isfinite([fc, fd]).all():
            raise FloatingPointError("Функция должна быть конечной")
        if b - a <= tolerance:
            x = (a + b) / 2
            return Minimum(x, float(function(x)), a, b, step, evaluations + 1)
        if step == max_iter:
            raise RuntimeError("Золотое сечение не достигло заданной точности")
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = float(function(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = float(function(d))
        evaluations += 1


def predict(x, theta, model):
    """Один вектор параметров или матрица векторов для разностных производных"""
    x, theta = np.asarray(x, float), np.asarray(theta, float)
    if model not in PARAMETERS or theta.ndim not in (1, 2):
        raise ValueError("Неизвестная модель или неверная форма параметров")
    if (
        theta.shape[-1] != len(PARAMETERS[model])
        or x.ndim != 1
        or not np.isfinite(x).all()
        or not np.isfinite(theta).all()
    ):
        raise ValueError("Неверные входы или параметры")
    p = [theta[..., j, None] for j in range(theta.shape[-1])]
    if model == "linear":
        return p[0] * x + p[1]
    if model == "quadratic":
        return p[0] * x**2 + p[1] * x + p[2]
    hidden1 = np.maximum(p[0] * x + p[1], 0)
    hidden2 = np.maximum(p[2] * x + p[3], 0)
    return np.maximum(p[4] * hidden1 + p[5] * hidden2 + p[6], 0)


def objective(theta, x, y, model, *, l2=0.0, scale=1.0):
    """scale * SSE + lambda * sum(weights**2); смещения не штрафуются"""
    error = predict(x, theta, model) - np.asarray(y, float)
    return scale * np.sum(error**2, axis=-1) + l2 * np.sum(
        np.asarray(theta) ** 2 * WEIGHT_MASK[model], axis=-1
    )


def numerical_gradient(function, theta, *, relative_step=1e-5):
    """Центральная разность. function принимает матрицу векторов параметров

    Вычисляются 2p значений скалярной целевой функции за один вызов NumPy
    h_j = relative_step * max(1, abs(theta_j)). Аналитических производных нет
    """
    theta = np.asarray(theta, float)
    if (
        theta.ndim != 1
        or not theta.size
        or not np.isfinite(theta).all()
        or not np.isfinite(relative_step)
        or relative_step <= 0
    ):
        raise ValueError("Неверные параметры численного дифференцирования")
    h = relative_step * np.maximum(1, np.abs(theta))
    offsets = np.diag(h)
    values = np.asarray(
        function(np.concatenate([theta + offsets, theta - offsets])), float
    )
    if values.shape != (2 * len(theta),) or not np.isfinite(values).all():
        raise FloatingPointError("Неверные значения функции в разностном градиенте")
    return (values[: len(theta)] - values[len(theta) :]) / (2 * h)


def initialize(model, seed=2026):
    """Одинаковая начальная точка для парных сравнений методов и штрафов"""
    if model not in PARAMETERS:
        raise ValueError("Неизвестная модель")
    rng = np.random.default_rng(seed)
    if model == "relu":
        # Оба скрытых узла имеют излом внутри диапазона данных; выход активен.
        return np.array([1.0, -0.35, -1.0, 1.3, 0.25, 0.25, 0.5]) + rng.uniform(
            -0.04, 0.04, 7
        )
    return rng.uniform(-0.1, 0.1, len(PARAMETERS[model]))


def minibatches(size, batch_size, rng):
    """Перемешивание без повторений: каждый объект ровно один раз за эпоху"""
    if (
        not isinstance(size, (int, np.integer))
        or size < 1
        or not isinstance(batch_size, (int, np.integer))
        or not 1 <= batch_size <= size
    ):
        raise ValueError("Неверный размер выборки или минибатча")
    order = rng.permutation(size)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


def optimizer_step(
    theta,
    gradient,
    first,
    second,
    step,
    *,
    method,
    learning_rate,
    beta=0.9,
    gamma=0.999,
    epsilon=1e-8,
):
    """Для Нестерова gradient уже вычислен в theta - learning_rate*beta*first"""
    if method in ("gd", "sgd"):
        return theta - learning_rate * gradient, first, second
    first = beta * first + (1 - beta) * gradient
    if method == "nesterov":
        return theta - learning_rate * first, first, second
    if method != "adam":
        raise ValueError("Неизвестный метод")
    second = gamma * second + (1 - gamma) * gradient**2
    first_hat = first / (1 - beta**step)
    second_hat = second / (1 - gamma**step)
    return (
        theta - learning_rate * first_hat / (np.sqrt(second_hat) + epsilon),
        first,
        second,
    )


@dataclass
class FitResult:
    theta: np.ndarray
    epochs: int
    updates: int
    converged: bool
    loss: float
    sse: float
    grad_norm: float
    weight_norm: float
    gradient_calls: int
    derivative_sample_evaluations: int
    history: np.ndarray


def train(
    x,
    y,
    model,
    *,
    method="gd",
    batch_size=None,
    learning_rate=0.01,
    max_epochs=3000,
    tolerance=1e-5,
    l2=0.0,
    seed=2026,
    initial=None,
    decay=300.0,
    relative_step=1e-5,
    beta=0.9,
    gamma=0.999,
    epsilon=1e-8,
    history_every=20,
):
    """Минимизировать SSE + L2, используя только центральные разности

    Для минибатча используется n/|B| * SSE_B + L2. Полный градиент проверяется
    после каждой эпохи. GD имеет постоянный шаг, другие методы - убывающий
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or not x.size or x.shape != y.shape or not np.isfinite([x, y]).all():
        raise ValueError("Нужны согласованные непустые конечные векторы x и y")
    if method not in ("gd", "sgd", "nesterov", "adam") or model not in PARAMETERS:
        raise ValueError("Неизвестная модель или метод")
    if (
        not np.isfinite(
            [learning_rate, tolerance, l2, decay, relative_step, beta, gamma, epsilon]
        ).all()
        or min(learning_rate, tolerance, decay, relative_step, epsilon) <= 0
        or l2 < 0
        or not 0 <= beta < 1
        or not 0 <= gamma < 1
        or not isinstance(max_epochs, (int, np.integer))
        or max_epochs < 0
        or not isinstance(history_every, (int, np.integer))
        or history_every < 1
    ):
        raise ValueError("Неверные параметры обучения")
    batch_size = len(x) if batch_size is None else batch_size
    if not isinstance(batch_size, (int, np.integer)) or not 1 <= batch_size <= len(x):
        raise ValueError("Неверный размер минибатча")
    if method == "gd" and batch_size != len(x):
        raise ValueError("Классический GD использует всю выборку")
    theta = (
        initialize(model, seed)
        if initial is None
        else np.array(initial, float, copy=True)
    )
    if theta.shape != (len(PARAMETERS[model]),) or not np.isfinite(theta).all():
        raise ValueError("Неверные начальные параметры")
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    rng = np.random.default_rng(seed)
    history, updates, gradient_calls, derivative_samples = [], 0, 0, 0
    full_objective = lambda t: objective(t, x, y, model, l2=l2)
    for epoch in range(max_epochs + 1):
        full_gradient = numerical_gradient(
            full_objective, theta, relative_step=relative_step
        )
        gradient_calls += 1
        derivative_samples += 2 * len(theta) * len(x)
        norm = float(np.max(np.abs(full_gradient)))
        sse = float(objective(theta, x, y, model))
        loss = float(full_objective(theta))
        weight_norm = float(np.sqrt(np.sum(theta**2 * WEIGHT_MASK[model])))
        converged = norm < tolerance
        lr = learning_rate if method == "gd" else learning_rate / (1 + epoch / decay)
        if epoch % history_every == 0 or converged or epoch == max_epochs:
            history.append(
                [epoch, updates, sse, loss, norm, weight_norm, lr, derivative_samples]
            )
        if converged or epoch == max_epochs:
            return FitResult(
                theta,
                epoch,
                updates,
                converged,
                loss,
                sse,
                norm,
                weight_norm,
                gradient_calls,
                derivative_samples,
                np.asarray(history),
            )
        batches = (
            [np.arange(len(x))]
            if batch_size == len(x)
            else minibatches(len(x), batch_size, rng)
        )
        for indices in batches:
            point = theta - lr * beta * first if method == "nesterov" else theta
            batch_objective = lambda t: objective(
                t, x[indices], y[indices], model, l2=l2, scale=len(x) / len(indices)
            )
            gradient = numerical_gradient(
                batch_objective, point, relative_step=relative_step
            )
            gradient_calls += 1
            derivative_samples += 2 * len(theta) * len(indices)
            updates += 1
            theta, first, second = optimizer_step(
                theta,
                gradient,
                first,
                second,
                updates,
                method=method,
                learning_rate=lr,
                beta=beta,
                gamma=gamma,
                epsilon=epsilon,
            )
            if not np.isfinite(theta).all():
                raise FloatingPointError("Обучение расходится: уменьшите шаг")


def save_model(path, model, result):
    np.savez_compressed(path, model=np.array(model), theta=result.theta)


def predict_saved(path, x):
    with np.load(Path(path), allow_pickle=False) as data:
        return predict(x, data["theta"], str(data["model"]))
