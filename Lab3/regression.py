"""МНК и t-интервалы"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURES = (
    "RNH2 R2NH R3N ROPO3 ROH RCHO RCOR RCOOH RCOOR ROR "
    "RSO2NR RSR RF RCl RBr RSO2R C RINGS AROMATIC HBA1 HBA2 HBD PSA MR"
).split()


def load_data(path=None):
    """Загрузка локального lipo.csv с проверкой столбцов и числовых значений"""
    path = Path(path) if path is not None else Path(__file__).with_name("lipo.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if (
        len(frame.columns) != 26
        or frame.columns[1] != "logP"
        or list(frame.columns[2:]) != FEATURES
    ):
        raise ValueError("Неожиданная схема lipo.csv")
    if frame.empty or frame.isna().any().any():
        raise ValueError("В данных есть пропуски или отсутствуют наблюдения")
    if not np.isfinite(frame.iloc[:, 1:].to_numpy(float)).all():
        raise ValueError("В данных есть неконечные значения")
    return frame


def design(x):
    """Добавление столбца единиц для свободного члена"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("X должна быть непустой конечной матрицей")
    return np.column_stack([np.ones(len(x)), x])


def metrics(y, pred):
    """Квадратичная ошибка и коэффициент детерминации"""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if (
        y.ndim != 1
        or y.size == 0
        or y.shape != pred.shape
        or not np.isfinite([y, pred]).all()
    ):
        raise ValueError("Ожидаются согласованные конечные векторы y и pred")
    mse = float(np.mean((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - len(y) * mse / sst) if sst > 0 else float("nan"),
    }


@dataclass
class OLSResult:
    coef: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    covariance: np.ndarray
    rank: int
    df: int
    sse: float
    s2: float
    r2: float

    def coefficients(self, names, confidence=0.95):
        """Оценки коэффициентов и индивидуальные интервалы Стьюдента"""
        if not 0 < confidence < 1 or len(names) + 1 != len(self.coef):
            raise ValueError("Неверные имена или доверительная вероятность")
        se = np.sqrt(np.diag(self.covariance))
        t = self.coef / se
        q = stats.t.ppf((1 + confidence) / 2, self.df)
        return pd.DataFrame(
            {
                "b": self.coef,
                "SE": se,
                "t": t,
                "p": 2 * stats.t.sf(abs(t), self.df),
                "CI_low": self.coef - q * se,
                "CI_high": self.coef + q * se,
            },
            index=["const", *names],
        )


def fit_ols(x, y):
    """Решение нормальных уравнений (A.T @ A) @ b = A.T @ y"""
    a, y = design(x), np.asarray(y, float)
    if y.shape != (len(a),) or not np.isfinite(y).all():
        raise ValueError("y должна соответствовать строкам X")
    rank = int(np.linalg.matrix_rank(a))
    if rank != a.shape[1] or len(a) <= rank:
        raise ValueError("Для МНК и t-интервалов нужны полный ранг и n > K")
    gram = a.T @ a
    b = np.linalg.solve(gram, a.T @ y)
    fitted = a @ b
    residuals = y - fitted
    sse = float(residuals @ residuals)
    df = len(a) - rank
    s2 = sse / df
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst == 0:
        raise ValueError("Дисперсия y должна быть положительной")
    covariance = s2 * np.linalg.solve(gram, np.eye(rank))
    return OLSResult(b, fitted, residuals, covariance, rank, df, sse, s2, 1 - sse / sst)
