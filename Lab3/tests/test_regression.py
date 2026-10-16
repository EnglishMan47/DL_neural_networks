import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regression import FEATURES, load_data, fit_ols, design


class RegressionTests(unittest.TestCase):
    def test_analytic_line_and_intervals(self):
        x = np.arange(5.0)[:, None]
        y = np.array([1, 2, 2, 4, 6.0])
        result = fit_ols(x, y)
        np.testing.assert_allclose(result.coef, [0.6, 1.2], atol=1e-13)
        self.assertAlmostEqual(result.sse, 1.6)
        self.assertAlmostEqual(result.s2, 1.6 / 3)
        expected_cov = (1.6 / 3) * np.array([[0.6, -0.2], [-0.2, 0.1]])
        np.testing.assert_allclose(result.covariance, expected_cov)
        table = result.coefficients(['x'])
        np.testing.assert_allclose(table.CI_high - table.b,
                                   stats.t.ppf(.975, 3) * np.sqrt(np.diag(expected_cov)))

    def test_local_data_and_normal_equations(self):
        frame = load_data()
        np.testing.assert_array_equal(frame.columns[2:], FEATURES)
        self.assertEqual(len(FEATURES), 24)
        expected = Path(__file__).resolve().parents[1] / 'lipo.csv'
        self.assertEqual(frame.to_csv(index=False), load_data(expected).to_csv(index=False))
        x, y = frame[FEATURES].to_numpy(float), frame.logP.to_numpy(float)
        result = fit_ols(x, y)
        np.testing.assert_allclose(design(x).T @ result.residuals, 0, atol=1e-8)
        self.assertEqual(result.rank, 25)
        self.assertEqual(result.df, 57)
        self.assertAlmostEqual(result.r2, 0.8379893589501719, places=10)
        self.assertAlmostEqual(result.s2, result.sse / 57)

    def test_rank_and_invalid_data(self):
        for x, y in [(np.ones((5, 2)), np.arange(5.)),
                     (np.array([[1.], [np.nan], [3.]]), np.arange(3.)),
                     (np.arange(4.)[:, None], np.arange(3.))]:
            with self.assertRaises(ValueError):
                fit_ols(x, y)


if __name__ == '__main__':
    unittest.main()
