"""Циклический метод вращений Якоби для вещественных симметричных матриц."""

from dataclasses import dataclass
import math
import numbers

import numpy as np


class JacobiError(Exception):
    """Базовая ошибка публичного интерфейса."""


class InvalidInputError(JacobiError, ValueError):
    """Неверная матрица или параметры алгоритма."""


class ConvergenceError(JacobiError):
    """Не удалось достичь точности за заданное число проходов."""


class NumericalError(JacobiError, ArithmeticError):
    """Задача выходит за поддерживаемые пределы арифметики float64."""


@dataclass(frozen=True)
class JacobiResult:
    """Собственные числа по возрастанию; соответствующие векторы в столбцах.

    offdiag_norm_scaled — норма внедиагональной части A_final / scale.
    tolerance_scaled — порог, с которым сравнивалась эта норма.
    history содержит эту норму до итераций и после каждого прохода.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    sweeps: int
    rotations: int
    scale: float
    offdiag_norm_scaled: float
    history: tuple[float, ...]
    tolerance_scaled: float


@dataclass(frozen=True)
class JacobiStep:
    """Результат одного вращения Якоби A' = P^T A P (формулы (9)–(15))."""

    matrix: np.ndarray
    tan: float
    cos: float
    sin: float


def _offdiag_norm(a):
    upper = a[np.triu_indices(a.shape[0], 1)]
    return math.sqrt(2.0) * math.hypot(*(float(x) for x in upper))


def _is_integer(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, numbers.Integral
    )


def _validate_matrix(matrix):
    try:
        raw = np.asarray(matrix)
        if raw.dtype.kind not in "iuf":
            raise InvalidInputError(
                "Элементы матрицы должны быть вещественными числами."
            )
        with np.errstate(over="raise", invalid="raise"):
            a = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise InvalidInputError(
            "Не удалось прочитать вещественную матрицу float64."
        ) from exc
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise InvalidInputError("Ожидается непустая квадратная матрица.")
    if not np.isfinite(a).all():
        raise InvalidInputError("Матрица не должна содержать NaN или бесконечности.")
    if not np.array_equal(a, a.T):
        raise InvalidInputError("Матрица должна быть симметричной: A = A.T.")
    return a


def _validate(matrix, epsilon, max_sweeps):
    if epsilon is not None:
        if isinstance(epsilon, (bool, np.bool_)) or not isinstance(
            epsilon, numbers.Real
        ):
            raise InvalidInputError(
                "epsilon должна быть положительным конечным числом."
            )
        try:
            epsilon = float(epsilon)
        except (ValueError, OverflowError) as exc:
            raise InvalidInputError("epsilon не представима как float64.") from exc
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise InvalidInputError(
                "epsilon должна быть положительным конечным числом."
            )
    if not _is_integer(max_sweeps) or max_sweeps < 0:
        raise InvalidInputError("max_sweeps должно быть целым неотрицательным числом.")
    return _validate_matrix(matrix), epsilon, int(max_sweeps)


def _rotation(app, aqq, apq):
    """tg, cos, sin угла, зануляющего apq != 0 (формулы (7)–(11))."""
    delta = aqq / 2 - app / 2
    # Устойчивая форма корня: t = sign(C)/(abs(C)+hypot(C,1)), C = delta/apq
    # Нормируем пару, чтобы не делить на очень малое apq
    local_scale = max(abs(delta), abs(apq))
    d, x = delta / local_scale, apq / local_scale
    if delta == 0:
        t = 1.0  # C = 0: допустимый поворот на pi/4
    else:
        t = x / (d + math.copysign(math.hypot(d, x), d))
    c = 1 / math.hypot(1, t)
    return t, c, t * c


def _rotate(b, v, p, q):
    """Зануляет b[p, q] на месте (формулы (12)–(15)), обновляет v по (17)–(18)."""
    apq = float(b[p, q])
    t, c, s = _rotation(float(b[p, p]), float(b[q, q]), apq)
    tau = s / (1 + c)
    mask = np.arange(b.shape[0])
    mask = mask[(mask != p) & (mask != q)]
    bp, bq = b[mask, p].copy(), b[mask, q].copy()
    b[mask, p] = bp - s * (bq + tau * bp)
    b[p, mask] = b[mask, p]
    b[mask, q] = bq + s * (bp - tau * bq)
    b[q, mask] = b[mask, q]
    b[p, p] -= apq * t
    b[q, q] += apq * t
    b[p, q] = b[q, p] = 0.0
    if v is not None:
        vp, vq = v[:, p].copy(), v[:, q].copy()
        v[:, p] = c * vp - s * vq
        v[:, q] = s * vp + c * vq
    return t, c, s


def jacobi_rotation(matrix, p, q):
    """Выполнить один шаг метода Якоби, зануляющий элемент (p, q).

    Индексы нумеруются с нуля, p != q. Входная матрица не изменяется.
    Если A[p, q] уже равен нулю, возвращается копия A и тождественный поворот.
    Функция предназначена для проверки формул метода на отдельном шаге;
    для полного разложения используйте jacobi_eigh.
    """
    a = _validate_matrix(matrix)
    n = a.shape[0]
    for index in (p, q):
        if not _is_integer(index) or not 0 <= index < n:
            raise InvalidInputError(
                f"Индексы должны быть целыми числами от 0 до {n - 1}."
            )
    if p == q:
        raise InvalidInputError("Зануляется только внедиагональный элемент: p != q.")
    if a[p, q] == 0:
        return JacobiStep(a, 0.0, 1.0, 0.0)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            t, c, s = _rotate(a, None, int(p), int(q))
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise NumericalError("Задача выходит за численные пределы float64.") from exc
    return JacobiStep(a, t, c, s)


def jacobi_eigh(matrix, *, epsilon=None, max_sweeps=100):
    """Найти собственные числа и ортонормированные собственные векторы.

    Parameters
    ----------
    matrix : array_like, shape (n, n)
        Непустая, точно симметричная вещественная матрица. Не изменяется.
    epsilon : float or None, default None
        Порог критерия остановки: sum(i != j, A_final[i,j]**2) < epsilon
        (абсолютная величина, сумма по ВСЕМ внедиагональным элементам).
        None — автоматический относительный порог: внедиагональная норма
        Фробениуса меньше machine_eps * ||A||_F. Так результат одинаково точен
        для матриц любого масштаба. Абсолютный epsilon не является
        относительной погрешностью собственных чисел.
    max_sweeps : int, default 100
        Максимальное число полных обходов верхнего треугольника.
        Ноль допустим: успех только для уже достаточно диагональной матрицы.

    Returns
    -------
    JacobiResult
        Сортированные собственные числа и соответствующие столбцы V.

    Raises
    ------
    InvalidInputError
        Нарушены предусловия.
    ConvergenceError
        Достигнут лимит проходов или возникла стагнация.
    NumericalError
        Переполнение, потеря ненулевых элементов при масштабировании
        или нарушение численных постусловий.

    Notes
    -----
    Только вращения Якоби: готовые eig/eigh внутри алгоритма не вызываются.
    Конечная точность float64 ограничивает точность результата даже при
    очень малом epsilon. Ошибки ОС, исчерпание памяти и прерывания пользователя
    не являются ошибками численного алгоритма и не перехватываются.
    """
    a, epsilon, max_sweeps = _validate(matrix, epsilon, max_sweeps)
    n = a.shape[0]
    scale = float(np.max(np.abs(a))) or 1.0
    with np.errstate(all="ignore"):
        b = a / scale
    if np.any((a != 0) & (b == 0)):
        raise NumericalError("Слишком большой разброс масштабов элементов float64.")
    original_scaled = b.copy()
    if epsilon is None:
        # ||offdiag(A)||F < eps_machine * ||A||F, в масштабированных единицах
        target = float(np.finfo(float).eps * np.linalg.norm(b, ord="fro"))
    else:
        # Эквивалентно S < epsilon, но без вычисления scale**2
        target = math.sqrt(epsilon) / scale
        if target == 0:
            raise NumericalError("Порог точности теряется при масштабировании.")
    v = np.eye(n)
    norm = _offdiag_norm(b)
    history = [norm]
    sweeps = rotations = 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            # norm > 0: уже диагональная (в т.ч. нулевая) матрица всегда готова
            while norm >= target and norm > 0:
                if sweeps >= max_sweeps:
                    raise ConvergenceError(
                        f"Точность не достигнута за {max_sweeps} проходов; "
                        f"норма/масштаб={norm:.3e}, порог/масштаб={target:.3e}."
                    )
                before = b.copy()
                for p in range(n - 1):
                    for q in range(p + 1, n):
                        if b[p, q] == 0:
                            continue
                        _rotate(b, v, p, q)
                        rotations += 1
                sweeps += 1
                norm = _offdiag_norm(b)
                history.append(norm)
                if norm >= target and np.array_equal(before, b):
                    raise ConvergenceError("Стагнация: точность недостижима в float64.")
            order = np.argsort(np.diag(b), kind="stable")
            scaled_values = np.diag(b)[order]
            values = scaled_values * scale
            v = v[:, order]
            # Детерминированный знак: наибольшая по модулю компонента положительна
            for j in range(n):
                if v[np.argmax(np.abs(v[:, j])), j] < 0:
                    v[:, j] *= -1
            residual = np.linalg.norm(
                original_scaled @ v - v * scaled_values, ord="fro"
            )
            orthogonality = np.linalg.norm(v.T @ v - np.eye(n), ord="fro")
            rounding = 128 * np.finfo(float).eps * n
            bound = norm + rounding * max(
                1.0, np.linalg.norm(original_scaled, ord="fro")
            )
            if (
                not np.isfinite(values).all()
                or not np.isfinite(v).all()
                or not math.isfinite(float(residual))
                or residual > bound
                or orthogonality > rounding
            ):
                raise NumericalError("Не выполнены численные постусловия разложения.")
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise NumericalError("Задача выходит за численные пределы float64.") from exc
    return JacobiResult(
        values, v, sweeps, rotations, scale, norm, tuple(history), target
    )
