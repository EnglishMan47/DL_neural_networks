"""Публичный интерфейс метода Якоби."""

from .solver import (
    ConvergenceError,
    InvalidInputError,
    JacobiError,
    JacobiResult,
    JacobiStep,
    NumericalError,
    jacobi_eigh,
    jacobi_rotation,
)

__version__ = "1.1.0"
__all__ = [
    "jacobi_eigh",
    "jacobi_rotation",
    "JacobiResult",
    "JacobiStep",
    "JacobiError",
    "InvalidInputError",
    "ConvergenceError",
    "NumericalError",
]
