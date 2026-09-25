"""Проверки генераторов ансамблей, спектральной статистики и аналитических ответов."""

from pathlib import Path
import math
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rmt  # noqa: E402


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_shapes_and_symmetry(self):
        for gen in [
            rmt.gaussian_ensemble,
            rmt.bernoulli_ensemble,
            rmt.bernoulli_sum_ensemble,
        ]:
            for n in [2, 4, 16]:
                with self.subTest(gen=gen.__name__, n=n):
                    m = gen(n, 50, self.rng)
                    self.assertEqual(m.shape, (50, n, n))
                    self.assertTrue(np.array_equal(m, np.swapaxes(m, 1, 2)))

    def test_gaussian_variances(self):
        # σ диагонали 2, внедиагональных элементов √2
        m = rmt.gaussian_ensemble(4, 40000, self.rng)
        diag = m[:, np.arange(4), np.arange(4)].ravel()
        off = m[:, 0, 1:].ravel()
        self.assertAlmostEqual(diag.mean(), 0, delta=0.03)
        self.assertAlmostEqual(diag.var(), 4, delta=0.1)
        self.assertAlmostEqual(off.var(), 2, delta=0.05)

    def test_bernoulli_values(self):
        m = rmt.bernoulli_ensemble(16, 2000, self.rng)
        self.assertTrue(np.all(np.abs(m) == 1))
        iu = np.triu_indices(16)
        upper = m[:, iu[0], iu[1]]
        self.assertAlmostEqual(np.mean(upper == 1), 0.5, delta=0.01)
        s = rmt.bernoulli_sum_ensemble(3, 2000, self.rng)
        self.assertTrue(set(np.unique(s)) <= {-2.0, 0.0, 2.0})

    def test_invalid_sizes(self):
        for n, count in [(1, 10), (2.5, 10), (2, 0)]:
            with self.subTest(n=n, count=count), self.assertRaises(ValueError):
                rmt.gaussian_ensemble(n, count, self.rng)


class SpectrumTests(unittest.TestCase):
    def test_sorted_and_jacobi_agrees(self):
        m = rmt.gaussian_ensemble(6, 30, np.random.default_rng(2))
        e = rmt.eigenvalues(m)
        self.assertTrue(np.all(np.diff(e, axis=1) >= 0))
        assert_allclose(rmt.eigenvalues(m, method="jacobi"), e, atol=1e-10)
        with self.assertRaises(ValueError):
            rmt.eigenvalues(m, method="other")

    def test_spacings_and_central_index(self):
        self.assertEqual([rmt.central_index(n) for n in (2, 4, 16)], [0, 1, 7])
        e = np.array([[0.0, 1.0, 3.0, 6.0], [1.0, 2.0, 2.0, 5.0]])
        assert_allclose(rmt.spacings(e), [2, 0])
        assert_allclose(rmt.spacings(e, 0), [1, 1])
        with self.assertRaises(ValueError):
            rmt.spacings(e, 3)

    def test_normalize(self):
        assert_allclose(rmt.normalize([1, 2, 3]), [0.5, 1, 1.5])
        with self.assertRaises(ValueError):
            rmt.normalize([0, 0])


class TheoryTests(unittest.TestCase):
    grid = np.linspace(0, 12, 200001)

    def integrate(self, f):
        return np.trapezoid(f, self.grid)

    def test_wigner_moments(self):
        p = rmt.wigner_pdf(self.grid)
        self.assertAlmostEqual(self.integrate(p), 1, places=8)
        self.assertAlmostEqual(self.integrate(self.grid * p), 1, places=8)
        var = self.integrate(self.grid**2 * p) - 1
        self.assertAlmostEqual(var, rmt.WIGNER_VARIANCE, places=8)
        assert_allclose(
            rmt.wigner_cdf(3.0), self.integrate(p * (self.grid <= 3.0)), atol=1e-4
        )

    def test_goe2_pdf_and_mean(self):
        grid = np.linspace(0, 80, 400001)
        p = rmt.goe2_raw_pdf(grid)
        self.assertAlmostEqual(np.trapezoid(p, grid), 1, places=8)
        self.assertAlmostEqual(
            np.trapezoid(grid * p, grid), 2 * math.sqrt(math.pi), places=7
        )
        # Масштабирование на среднее 2√π даёт в точности догадку Вигнера
        s = np.linspace(0, 4, 9)
        c = 2 * math.sqrt(math.pi)
        assert_allclose(c * rmt.goe2_raw_pdf(c * s), rmt.wigner_pdf(s), atol=1e-14)

    def test_goe2_sampling_matches_theory(self):
        rng = np.random.default_rng(3)
        raw = rmt.spacings(rmt.eigenvalues(rmt.gaussian_ensemble(2, 40000, rng)))
        self.assertAlmostEqual(raw.mean(), 2 * math.sqrt(math.pi), delta=0.04)
        self.assertLess(rmt.ks_statistic(rmt.normalize(raw), rmt.wigner_cdf), 0.01)

    def test_ks_and_kl(self):
        rng = np.random.default_rng(4)
        u = rng.random(20000)
        wigner_sample = np.sqrt(-4 * np.log1p(-u) / np.pi)  # обратная функция F
        self.assertLess(rmt.ks_statistic(wigner_sample, rmt.wigner_cdf), 0.015)
        self.assertGreater(rmt.ks_statistic(wigner_sample, rmt.poisson_cdf), 0.15)
        bins = np.linspace(0, 4, 41)
        self.assertLess(rmt.binned_kl(wigner_sample, rmt.wigner_cdf, bins), 0.005)
        self.assertGreater(
            rmt.binned_kl(rng.exponential(size=20000), rmt.wigner_cdf, bins), 0.2
        )


class ExactResultTests(unittest.TestCase):
    def test_enumeration(self):
        m = rmt.enumerate_symmetric((-1, 1), 3)
        self.assertEqual(m.shape, (64, 3, 3))
        self.assertEqual(len({x.tobytes() for x in m}), 64)

    def test_bernoulli_2x2_analytic(self):
        values, probs, mean = rmt.exact_spacing_distribution((-1, 1), 2)
        tv, tp, tm = rmt.bernoulli_2x2_theory()
        assert_allclose(values, tv, atol=1e-9)
        assert_allclose(probs, tp)
        self.assertAlmostEqual(mean, tm, places=12)
        self.assertAlmostEqual(float(np.dot(tv, tp)), 1, places=12)

    def test_bernoulli_sum_2x2_analytic(self):
        # Перебор A (4 элемента +-1) и симметризация B = A + A^T
        combos = (
            np.array(np.meshgrid(*[[-1.0, 1.0]] * 4)).reshape(4, -1).T.reshape(-1, 2, 2)
        )
        s = rmt.spacings(rmt.eigenvalues(combos + np.swapaxes(combos, 1, 2)))
        s[s < 1e-9] = 0
        tv, tp, tm = rmt.bernoulli_sum_2x2_theory()
        self.assertAlmostEqual(s.mean(), tm, places=12)
        values, counts = np.unique(np.round(s / s.mean(), 9), return_counts=True)
        assert_allclose(values, tv, atol=1e-9)
        assert_allclose(counts / counts.sum(), tp)

    def test_bernoulli_4x4_degeneracy(self):
        values, probs, _ = rmt.exact_spacing_distribution((-1, 1), 4)
        self.assertAlmostEqual(float(np.dot(values, probs)), 1, places=9)
        self.assertEqual(values[0], 0)
        self.assertAlmostEqual(probs[0], 9 / 64, places=12)


if __name__ == "__main__":
    unittest.main()
