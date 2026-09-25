"""Проверки по аналитическим ответам, инвариантам и независимому эталону."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from jacobi_lab import (jacobi_eigh, jacobi_rotation, InvalidInputError, ConvergenceError,
                        NumericalError)


class JacobiTests(unittest.TestCase):
    def check_decomposition(self, a, **kwargs):
        a = np.array(a, dtype=float)
        saved = a.copy()
        r = jacobi_eigh(a, **kwargs)
        scale = np.max(np.abs(a)) or 1.0
        b, w = a / scale, r.eigenvalues / scale
        assert_allclose(b @ r.eigenvectors, r.eigenvectors * w, atol=1e-9, rtol=1e-9)
        assert_allclose(r.eigenvectors.T @ r.eigenvectors, np.eye(len(a)), atol=1e-12)
        assert_allclose(r.eigenvectors @ np.diag(w) @ r.eigenvectors.T, b, atol=1e-9)
        assert_allclose(w, np.linalg.eigvalsh(b), atol=1e-9, rtol=1e-9)
        self.assertTrue(np.array_equal(a, saved))
        self.assertTrue(np.all(np.diff(r.eigenvalues) >= 0))
        if kwargs.get("epsilon") is not None:
            self.assertEqual(r.tolerance_scaled, np.sqrt(kwargs["epsilon"]) / r.scale)
        self.assertTrue(r.offdiag_norm_scaled < r.tolerance_scaled or r.offdiag_norm_scaled == 0)
        return r

    def test_scalar(self):
        self.check_decomposition([[-7]])

    def test_zero(self):
        r = self.check_decomposition(np.zeros((4, 4)))
        self.assertEqual(r.rotations, 0)

    def test_diagonal_sorted(self):
        r = self.check_decomposition(np.diag([3, -1, 2]))
        assert_allclose(r.eigenvalues, [-1, 2, 3])
        self.assertEqual(r.sweeps, 0)

    def test_equal_diagonal(self):
        r = self.check_decomposition([[2, 1], [1, 2]])
        assert_allclose(r.eigenvalues, [1, 3])

    def test_negative_offdiagonal(self):
        self.check_decomposition([[2, -1], [-1, 2]])

    def test_indefinite(self):
        self.check_decomposition([[0, 2, -3], [2, 0, 4], [-3, 4, 1]])

    def test_repeated_spectrum(self):
        self.check_decomposition(np.eye(6) + np.ones((6, 6)))

    def test_lab_example(self):
        a = [[1 + 3 * (i + j) for j in range(5)] for i in range(5)]
        exact = [(65 - np.sqrt(6025)) / 2, 0, 0, 0, (65 + np.sqrt(6025)) / 2]
        for eps in [None, 1e-20]:
            with self.subTest(epsilon=eps):
                r = self.check_decomposition(a, epsilon=eps)
                assert_allclose(r.eigenvalues, exact, atol=1e-10)

    def test_lab_single_rotation_example(self):
        # Численный пример: зануление A24 (индексы с нуля: 1, 3).
        a = [[1 + 3 * (i + j) for j in range(5)] for i in range(5)]
        step = jacobi_rotation(a, 1, 3)
        self.assertAlmostEqual(step.tan, 13 / (6 + np.sqrt(205)), places=14)
        assert_allclose([step.tan, step.cos, step.sin], [0.64, 0.842, 0.539], atol=5e-3)
        expected = [[1, -2.02, 7, 10.58, 13],
                    [-2.02, -1.32, -0.2, 0, 1.62],
                    [7, -0.2, 13, 18.86, 19],
                    [10.58, 0, 18.86, 27.32, 27.15],
                    [13, 1.62, 19, 27.15, 25]]
        # В эталоне вычисления округлены до tg = 0.64, отсюда допуск 0.02.
        assert_allclose(step.matrix, expected, atol=2e-2)
        self.assertEqual(step.matrix[1, 3], 0)
        self.assertTrue(np.array_equal(step.matrix, step.matrix.T))
        p = np.eye(5)
        p[1, 1] = p[3, 3] = step.cos
        p[1, 3], p[3, 1] = step.sin, -step.sin
        assert_allclose(step.matrix, p.T @ np.array(a, float) @ p, atol=1e-12)

    def test_rotation_contract(self):
        step = jacobi_rotation(np.diag([1., 2.]), 0, 1)
        self.assertEqual((step.tan, step.cos, step.sin), (0.0, 1.0, 0.0))
        for p, q in [(0, 0), (0, 2), (-1, 0), (0.0, 1), (True, 0), ("0", 1)]:
            with self.subTest(p=p, q=q), self.assertRaises(InvalidInputError):
                jacobi_rotation([[1, 2], [2, 1]], p, q)

    def test_small_scale_default(self):
        # Абсолютный порог 1e-20 здесь выполнен сразу, без вращений, и дал бы
        # неверный ответ [2e-12, 2e-12]; порог по умолчанию относительный.
        for s in [1e-12, 1e-100, 1e100]:
            with self.subTest(scale=s):
                r = self.check_decomposition(np.array([[2., 1.], [1., 2.]]) * s)
                assert_allclose(r.eigenvalues, [s, 3 * s], rtol=1e-14)

    def test_random_symmetric(self):
        rng = np.random.default_rng(42)
        for n in [2, 3, 5, 10, 20]:
            for _ in range(4):
                with self.subTest(n=n):
                    x = rng.normal(size=(n, n))
                    self.check_decomposition(x + x.T)

    def test_hilbert(self):
        i = np.arange(8)
        self.check_decomposition(1 / (i[:, None] + i[None, :] + 1))

    def test_extreme_scales(self):
        self.check_decomposition(np.array([[2., 1.], [1., 2.]]) * 1e150, epsilon=1e280)
        self.check_decomposition(np.array([[2., 1.], [1., 2.]]) * 1e-150, epsilon=1e-320)

    def test_readonly_input(self):
        a = np.array([[2., 1.], [1., 2.]])
        a.setflags(write=False)
        jacobi_eigh(a)

    def test_limit(self):
        with self.assertRaises(ConvergenceError):
            jacobi_eigh([[2, 1], [1, 2]], max_sweeps=0)
        jacobi_eigh(np.eye(2), max_sweeps=0)

    def test_one_sweep_insufficient(self):
        with self.assertRaises(ConvergenceError):
            jacobi_eigh([[1, 2, 3], [2, 4, 5], [3, 5, 6]], max_sweeps=1)

    def test_invalid_matrices(self):
        for a in [[], [1, 2], [[1, 2]], [[1], [2, 3]], [[1, 2], [3, 4]],
                  [[np.nan]], [[np.inf]], [[1j]], [["1"]], [[True]], None]:
            with self.subTest(a=a), self.assertRaises(InvalidInputError):
                jacobi_eigh(a)

    def test_invalid_epsilon(self):
        for eps in [0, -1, np.nan, np.inf, True, "1e-9", 10**400]:
            with self.subTest(epsilon=eps), self.assertRaises(InvalidInputError):
                jacobi_eigh([[1]], epsilon=eps)

    def test_invalid_limit(self):
        for limit in [-1, 1.2, True, "10", None]:
            with self.subTest(limit=limit), self.assertRaises(InvalidInputError):
                jacobi_eigh([[1]], max_sweeps=limit)

    def test_overflow_is_expected_error(self):
        with self.assertRaises(NumericalError):
            jacobi_eigh([[1.7e308, 1.7e308], [1.7e308, 1.7e308]])

    def test_scale_loss_is_expected_error(self):
        with self.assertRaises(NumericalError):
            jacobi_eigh([[1e308, 1e-308], [1e-308, 1]])

    def test_default_random_many(self):
        rng = np.random.default_rng(7)
        for n in [4, 16, 40]:
            for s in [1e-200, 1e-8, 1, 1e8, 1e200]:
                with self.subTest(n=n, scale=s):
                    x = rng.normal(size=(n, n))
                    self.check_decomposition((x + x.T) * s)

    def test_full_offdiagonal_criterion(self):
        # Сумма по верхнему треугольнику = 1, полная сумма = 2.
        r = jacobi_eigh([[2, 1], [1, 2]], epsilon=1.5)
        self.assertEqual(r.rotations, 1)

    def test_convergence_history(self):
        r = jacobi_eigh([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
        self.assertEqual(len(r.history), r.sweeps + 1)
        self.assertTrue(all(y <= x + 1e-14 for x, y in zip(r.history, r.history[1:])))


if __name__ == "__main__":
    unittest.main()
