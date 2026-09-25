"""Запуск: python -m jacobi_lab [matrix.json] [--epsilon 1e-20] [--max-sweeps 100]."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import JacobiError, jacobi_eigh


def main():
    parser = argparse.ArgumentParser(
        description="Собственные числа и векторы: метод Якоби"
    )
    parser.add_argument(
        "matrix", nargs="?", type=Path, help="JSON-файл с квадратной матрицей"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="абсолютный порог суммы квадратов внедиагональных элементов; "
        "по умолчанию — автоматический относительный порог",
    )
    parser.add_argument("--max-sweeps", type=int, default=100)
    args = parser.parse_args()
    try:
        matrix = (
            json.loads(args.matrix.read_text(encoding="utf-8-sig"))
            if args.matrix
            else [[1 + 3 * (i + j) for j in range(5)] for i in range(5)]
        )
        result = jacobi_eigh(matrix, epsilon=args.epsilon, max_sweeps=args.max_sweeps)
    except (JacobiError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    np.set_printoptions(precision=10, suppress=True, linewidth=100)
    print("Метод Якоби")
    print("Собственные числа:\n", result.eigenvalues)
    print("Собственные векторы (столбцы):\n", result.eigenvectors)
    print(f"Проходов: {result.sweeps}; вращений: {result.rotations}")
    print(
        f"Норма внедиагональной части / масштаб: {result.offdiag_norm_scaled:.3e} "
        f"(порог {result.tolerance_scaled:.3e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
