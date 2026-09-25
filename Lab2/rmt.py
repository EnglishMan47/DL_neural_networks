"""Лабораторная работа 2: Распределение расстояний между собственными числами
случайных симметричных матриц.

Модуль содержит генераторы ансамблей, вычисление упорядоченных спектров,
расстояний между фиксированной парой соседних собственных чисел, теоретические
распределения (догадка Вигнера, пуассоновское) и меры согласия.
"""

from itertools import product
import math
from pathlib import Path
import sys

import numpy as np

# Точные значения для догадки Вигнера: <s> = 1, D[s] = 4/pi - 1
WIGNER_VARIANCE = 4 / math.pi - 1


# ---------------------------------------------------------------- ансамбли
def gaussian_ensemble(n, count, rng, sigma=1.0):
    """Гауссов ортогональный ансамбль по формуле: B = A + A^T.

    Элементы A независимы и распределены N(0, sigma^2). После симметризации
    среднеквадратичное отклонение диагональных элементов равно 2·sigma,
    внедиагональных — √2·sigma.
    Возвращает массив формы (count, n, n).
    """
    _check_size(n, count)
    a = rng.normal(0.0, sigma, size=(count, n, n))
    return a + np.swapaxes(a, 1, 2)


def bernoulli_ensemble(n, count, rng):
    """Симметричные матрицы, все элементы которых равны −1 или 1 с вероятностью 1/2.

    Независимы элементы верхнего треугольника вместе с диагональю.
    """
    _check_size(n, count)
    a = rng.choice(np.array([-1.0, 1.0]), size=(count, n, n))
    return np.triu(a) + np.swapaxes(np.triu(a, 1), 1, 2)


def bernoulli_sum_ensemble(n, count, rng):
    """Симметризация B = A + A^T матрицы с элементами +-1.
    Элементы B принимают значения −2, 0, 2."""
    _check_size(n, count)
    a = rng.choice(np.array([-1.0, 1.0]), size=(count, n, n))
    return a + np.swapaxes(a, 1, 2)


def _check_size(n, count):
    if int(n) != n or n < 2:
        raise ValueError("Размер матрицы n должен быть целым и не меньше 2.")
    if int(count) != count or count < 1:
        raise ValueError("Размер ансамбля должен быть положительным целым.")


# ---------------------------------------------------------------- спектры
def _jacobi_eigh():
    lab1 = Path(__file__).resolve().parent.parent / "Lab1"
    if (lab1 / "jacobi_lab").is_dir() and str(lab1) not in sys.path:
        sys.path.insert(0, str(lab1))
    from jacobi_lab import jacobi_eigh

    return jacobi_eigh


def eigenvalues(matrices, method="numpy"):
    """Собственные числа каждой матрицы ансамбля по возрастанию, форма (count, n).

    method="numpy" — векторизованный numpy.linalg.eigvalsh;
    method="jacobi" — метод Якоби для проверки.
    """
    matrices = np.asarray(matrices, dtype=float)
    if method == "numpy":
        return np.linalg.eigvalsh(matrices)
    if method == "jacobi":
        jacobi_eigh = _jacobi_eigh()
        return np.array([jacobi_eigh(m).eigenvalues for m in matrices])
    raise ValueError("method должен быть 'numpy' или 'jacobi'.")


def central_index(n):
    """Номер k (с нуля) нижнего собственного числа центральной пары (λ_k, λ_k+1).

    В центре спектра плотность собственных чисел почти постоянна, поэтому
    нормированные расстояния здесь лучше всего описываются догадкой Вигнера.
    Для n = 2, 4, 16 это k = 0, 1, 7.
    """
    return max(n // 2 - 1, 0)


def spacings(eigs, k=None):
    """Расстояния λ_{k+1} − λ_k для фиксированной пары во всех матрицах ансамбля.

    eigs — массив (count, n) упорядоченных собственных чисел.
    Пара одна и та же для всех матриц; по умолчанию центральная.
    """
    eigs = np.asarray(eigs, dtype=float)
    n = eigs.shape[1]
    k = central_index(n) if k is None else k
    if not 0 <= k < n - 1:
        raise ValueError(f"k должно лежать в диапазоне 0..{n - 2}.")
    return eigs[:, k + 1] - eigs[:, k]


def normalize(s):
    """Нормировка расстояний на среднее значение."""
    s = np.asarray(s, dtype=float)
    mean = s.mean()
    if not mean > 0:
        raise ValueError("Среднее расстояние должно быть положительным.")
    return s / mean


# ---------------------------------------------------------------- теория
def wigner_pdf(s):
    """Догадка Вигнера: ρ(s) = (πs/2)·exp(−πs²/4)."""
    s = np.asarray(s, dtype=float)
    return np.pi * s / 2 * np.exp(-np.pi * s**2 / 4)


def wigner_cdf(s):
    """Функция распределения догадки Вигнера: F(s) = 1 − exp(−πs²/4)."""
    s = np.asarray(s, dtype=float)
    return -np.expm1(-np.pi * s**2 / 4)


def goe2_raw_pdf(s):
    """Плотность ненормированного расстояния для GOE 2×2 при σ = 1."""
    s = np.asarray(s, dtype=float)
    return s / 8 * np.exp(-(s**2) / 16)


def poisson_pdf(s):
    """Пуассоновская статистика (некоррелированные уровни): ρ(s) = exp(−s)."""
    return np.exp(-np.asarray(s, dtype=float))


def poisson_cdf(s):
    return -np.expm1(-np.asarray(s, dtype=float))


# ---------------------------------------------------------------- согласие
def ks_statistic(sample, cdf):
    """Статистика Колмогорова–Смирнова D = sup |F_эмп(s) − F(s)|."""
    x = np.sort(np.asarray(sample, dtype=float))
    f = cdf(x)
    i = np.arange(1, x.size + 1)
    return float(max(np.max(i / x.size - f), np.max(f - (i - 1) / x.size)))


def binned_kl(sample, cdf, bins):
    """Расхождение Кульбака–Лейблера D_KL(эмпирическое ‖ теория) по бинам гистограммы.

    Вероятность бина по теории вычисляется точно через функцию распределения;
    последний бин включает весь хвост до бесконечности.
    """
    bins = np.asarray(bins, dtype=float)
    counts, _ = np.histogram(np.clip(sample, bins[0], bins[-1]), bins=bins)
    p = counts / counts.sum()
    edges = cdf(bins)
    q = np.diff(edges)
    q[-1] += 1 - edges[-1]
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def describe(s):
    """Моменты нормированных расстояний и доля малых расстояний."""
    s = np.asarray(s, dtype=float)
    return {
        "mean": float(s.mean()),
        "variance": float(s.var()),
        "p_below_0.1": float(np.mean(s < 0.1)),
        "p_zero": float(np.mean(s < 1e-9)),
    }


# ---------------------------------------------------------------- точные результаты
def enumerate_symmetric(values, n):
    """Все симметричные матрицы n×n, независимые элементы которых (верхний
    треугольник с диагональю) пробегают values. Возвращает массив (m, n, n).

    Для values = (−1, 1) и n = 4 это 2^10 = 1024 равновероятные матрицы.
    """
    iu = np.triu_indices(n)
    free = len(iu[0])
    combos = np.array(list(product(values, repeat=free)), dtype=float)
    out = np.zeros((len(combos), n, n))
    out[:, iu[0], iu[1]] = combos
    out[:, iu[1], iu[0]] = combos
    return out


def exact_spacing_distribution(values, n, k=None, decimals=9):
    """Точное распределение нормированного расстояния для равновероятного перебора.

    Возвращает (значения, вероятности, среднее ненормированное расстояние).
    Значения округляются до decimals знаков, чтобы объединить совпадающие.
    """
    s = spacings(eigenvalues(enumerate_symmetric(values, n)), k)
    s[np.abs(s) < 10.0**-decimals] = 0.0
    mean = s.mean()
    uniq, counts = np.unique(np.round(s / mean, decimals), return_counts=True)
    return uniq, counts / counts.sum(), mean


def bernoulli_2x2_theory():
    """Аналитический ответ для матриц [[a, b], [b, c]], a, b, c = +-1.

    s = √((c − a)² + 4b²) = 2 или 2√2 с вероятностью 1/2; <s> = 1 + √2.
    После нормировки: 2(√2 − 1) ≈ 0.828 и 4 − 2√2 ≈ 1.172, по 1/2.
    """
    r2 = math.sqrt(2)
    return np.array([2 * (r2 - 1), 4 - 2 * r2]), np.array([0.5, 0.5]), 1 + r2


def bernoulli_sum_2x2_theory():
    """Аналитический ответ для B = A + A^T, A_ij = +-1 (элементы B ∈ {−2, 0, 2}).

    s = 0, 4, 4√2 с вероятностями 1/4, 1/2, 1/4; <s> = 2 + √2.
    После нормировки: 0, 2(2 − √2) ≈ 1.172, 4√2 − 4 ≈ 1.657.
    """
    r2 = math.sqrt(2)
    return (
        np.array([0.0, 2 * (2 - r2), 4 * r2 - 4]),
        np.array([0.25, 0.5, 0.25]),
        2 + r2,
    )
