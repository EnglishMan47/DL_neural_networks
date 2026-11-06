import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import softmax_jacobian, stable_softmax


LOGITS = [-3, 1, 100, 5, -1]


class SoftmaxTests(unittest.TestCase):
    def test_all_25_derivatives_by_exponentials(self):
        exponentials = np.exp(np.array(LOGITS, dtype=float))
        denominator = exponentials.sum() ** 2
        expected = np.empty((5, 5))
        for i in range(5):
            for j in range(5):
                if i == j:
                    # Диагональ считаем без вычитания близких чисел
                    others = sum(exponentials[k] for k in range(5) if k != i)
                    expected[i, j] = exponentials[i] * others / denominator
                else:
                    expected[i, j] = -exponentials[i] * exponentials[j] / denominator
        np.testing.assert_allclose(softmax_jacobian(LOGITS), expected,
                                   rtol=1e-5, atol=0)

    def test_probabilities_to_five_significant_digits(self):
        exponentials = np.exp(np.array(LOGITS, dtype=float))
        expected = exponentials / exponentials.sum()
        probabilities = stable_softmax(LOGITS)
        np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-7)
        self.assertEqual([f'{value:.5g}' for value in probabilities],
                         ['1.8521e-45', '1.0112e-43', '1', '5.5211e-42', '1.3685e-44'])

    def test_small_diagonal_value(self):
        jacobian = softmax_jacobian(LOGITS)
        self.assertGreater(jacobian[2, 2], 0.)
        np.testing.assert_allclose(jacobian[2, 2], 5.6377e-42, rtol=1e-5, atol=0)

    def test_derivatives_for_ordinary_logits(self):
        values = np.array([-1., .2, .6, 1.3, -.4])
        probabilities = np.exp(values) / np.exp(values).sum()
        expected = np.diag(probabilities) - np.outer(probabilities, probabilities)
        np.testing.assert_allclose(softmax_jacobian(values), expected,
                                   rtol=1e-5, atol=1e-7)

    def test_invalid_logits_and_axes(self):
        for values in ([], [np.nan], [np.inf], np.zeros((2, 2, 2)), np.empty((2, 0))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stable_softmax(values)
        for axis in (1, -2, .5, 'columns'):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                stable_softmax([1., 2.], axis=axis)
        with self.assertRaises(ValueError):
            softmax_jacobian([[1., 2.]])


if __name__ == '__main__':
    unittest.main()
