"""Самовнимание, подсчёт параметров и производные softmax"""

from dataclasses import dataclass

import numpy as np


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} должно быть положительным целым числом")
    return int(value)


def parameter_counts(dimension, tokens, hidden=None):
    """Скрытый слой перцептрона по умолчанию имеет ширину D*N"""
    d = _positive_integer(dimension, "D")
    n = _positive_integer(tokens, "N")
    h = d * n if hidden is None else _positive_integer(hidden, "H")
    return {
        "D": d,
        "N": n,
        "H": h,
        "QKV_weights": 3 * d * d,
        "QKV_biases": 3 * d,
        "QKV_parameters": 3 * d * d + 3 * d,
        "attention_entries": n * n,
        "attention_extra_parameters": 0,
        "MLP_parameters": 2 * d * n * h + h + d * n,
    }


def stable_softmax(z, axis=0):
    """Нормировка по заданной оси с вычитанием максимума"""
    z = np.asarray(z, dtype=float)
    if z.ndim not in (1, 2) or not z.size or 0 in z.shape or not np.isfinite(z).all():
        raise ValueError("Нужен непустой конечный вектор или матрица")
    if not isinstance(axis, (int, np.integer)) or not -z.ndim <= axis < z.ndim:
        raise ValueError("Неверная ось softmax")
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def softmax_jacobian(z):
    """J[i,j] = dy_i/dz_j. Диагональ вычисляется без вычитания из единицы"""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("Для якобиана нужен вектор логитов")
    probabilities = stable_softmax(z)
    jacobian = -np.outer(probabilities, probabilities)
    for i in range(len(z)):
        # При насыщении 1-y_i может округлиться до нуля
        other = np.sum(probabilities[:i]) + np.sum(probabilities[i + 1 :])
        jacobian[i, i] = probabilities[i] * other
    return jacobian


@dataclass
class AttentionParameters:
    w_v: np.ndarray
    b_v: np.ndarray
    w_k: np.ndarray
    b_k: np.ndarray
    w_q: np.ndarray
    b_q: np.ndarray


@dataclass
class AttentionResult:
    values: np.ndarray
    keys: np.ndarray
    queries: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    output: np.ndarray


def initialize(dimension, seed=2026):
    d = _positive_integer(dimension, "D")
    rng = np.random.default_rng(seed)
    return AttentionParameters(
        rng.uniform(-0.5, 0.5, (d, d)),
        rng.uniform(-0.1, 0.1, d),
        rng.uniform(-0.5, 0.5, (d, d)),
        rng.uniform(-0.1, 0.1, d),
        rng.uniform(-0.5, 0.5, (d, d)),
        rng.uniform(-0.1, 0.1, d),
    )


def self_attention(x, parameters, *, scaled=True):
    """Векторы расположены в столбцах X; каждый столбец внимания нормирован по ключам"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not x.size or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError("X должна быть непустой конечной матрицей D x N")
    d = x.shape[0]
    projections = []
    for weights, biases in (
        (parameters.w_v, parameters.b_v),
        (parameters.w_k, parameters.b_k),
        (parameters.w_q, parameters.b_q),
    ):
        weights, biases = np.asarray(weights, float), np.asarray(biases, float)
        if weights.shape != (d, d) or biases.shape != (d,):
            raise ValueError("Неверные размеры весов или смещений")
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError("Веса и смещения должны быть конечными")
        projections.append(weights @ x + biases[:, None])
    values, keys, queries = projections
    scores = keys.T @ queries
    if scaled:
        scores /= np.sqrt(d)
    weights = stable_softmax(scores, axis=0)
    return AttentionResult(values, keys, queries, scores, weights, values @ weights)


def permutation_matrix(order):
    """X @ P переставляет столбцы X в порядке order"""
    order = np.asarray(order)
    if (
        order.ndim != 1
        or not order.size
        or order.dtype.kind not in "iu"
        or not np.array_equal(np.sort(order), np.arange(len(order)))
    ):
        raise ValueError("Нужна перестановка индексов от 0 до N-1")
    return np.eye(len(order))[:, order]


