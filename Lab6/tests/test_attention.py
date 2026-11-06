import math
import sys
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import (initialize, parameter_counts, permutation_matrix,
                       self_attention)


class AttentionTests(unittest.TestCase):
    def test_parameter_counts_against_allocated_arrays(self):
        for d in (1, 3, 8):
            params = initialize(d)
            weights = (params.w_v, params.w_k, params.w_q)
            biases = (params.b_v, params.b_k, params.b_q)
            for n in (1, 4, 9):
                with self.subTest(D=d, N=n):
                    counts = parameter_counts(d, n)
                    self.assertEqual(counts['QKV_weights'], sum(a.size for a in weights))
                    self.assertEqual(counts['QKV_biases'], sum(a.size for a in biases))
                    self.assertEqual(counts['QKV_parameters'], sum(a.size for a in vars(params).values()))
                    result = self_attention(np.ones((d, n)), params)
                    self.assertEqual(counts['attention_entries'], result.weights.size)
                    self.assertEqual(counts['attention_extra_parameters'], 0)

    def test_mlp_parameter_count_for_general_hidden_width(self):
        for d, n, h in ((2, 4, 3), (3, 5, 7), (1, 1, 1)):
            arrays = (np.empty((h, d*n)), np.empty(h),
                      np.empty((d*n, h)), np.empty(d*n))
            counts = parameter_counts(d, n, hidden=h)
            self.assertEqual(counts['H'], h)
            self.assertEqual(counts['MLP_parameters'], sum(a.size for a in arrays))
        default = parameter_counts(3, 4)
        self.assertEqual(default['H'], 12)
        self.assertEqual(default['MLP_parameters'], 312)

    def test_permutation_matrix_from_assignment(self):
        matrix = permutation_matrix([1, 3, 0, 2])
        expected = np.array([[0, 0, 1, 0], [1, 0, 0, 0],
                             [0, 0, 0, 1], [0, 1, 0, 0]])
        np.testing.assert_array_equal(matrix, expected)
        x = np.arange(12).reshape(3, 4)
        np.testing.assert_array_equal(x @ matrix, x[:, [1, 3, 0, 2]])
        np.testing.assert_array_equal(matrix.T @ matrix, np.eye(4))

    def test_attention_against_independent_query_loop(self):
        x = np.array([[.2, 1., -.5, .7], [1.2, -.3, .8, .1], [-.4, .6, 1.1, -.2]])
        original = x.copy()
        params = initialize(3)
        before = {name: value.copy() for name, value in vars(params).items()}
        values = [params.w_v @ x[:, i] + params.b_v for i in range(4)]
        keys = [params.w_k @ x[:, i] + params.b_k for i in range(4)]
        queries = [params.w_q @ x[:, i] + params.b_q for i in range(4)]
        for scaled in (False, True):
            with self.subTest(scaled=scaled):
                expected = np.empty_like(x)
                expected_weights = np.empty((4, 4))
                for j, query in enumerate(queries):
                    scores = [float(key @ query) / (math.sqrt(3) if scaled else 1) for key in keys]
                    exps = [math.exp(score - max(scores)) for score in scores]
                    probabilities = [value / sum(exps) for value in exps]
                    expected_weights[:, j] = probabilities
                    expected[:, j] = sum(probability * value for probability, value in zip(probabilities, values))
                actual = self_attention(x, params, scaled=scaled)
                np.testing.assert_allclose(actual.output, expected, atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(actual.weights, expected_weights, atol=1e-7, rtol=1e-5)
        np.testing.assert_array_equal(x, original)
        for name, value in vars(params).items():
            np.testing.assert_array_equal(value, before[name])

    def test_all_24_permutations_with_and_without_scaling(self):
        x = np.random.default_rng(42).normal(size=(3, 4))
        params = initialize(3)
        for scaled in (False, True):
            base = self_attention(x, params, scaled=scaled)
            for order in permutations(range(4)):
                with self.subTest(scaled=scaled, order=order):
                    p = permutation_matrix(order)
                    actual = self_attention(x @ p, params, scaled=scaled)
                    for field in ('values', 'keys', 'queries', 'output'):
                        np.testing.assert_allclose(getattr(actual, field), getattr(base, field) @ p,
                                                   atol=1e-7, rtol=1e-5)
                    for field in ('scores', 'weights'):
                        np.testing.assert_allclose(getattr(actual, field), p.T @ getattr(base, field) @ p,
                                                   atol=1e-7, rtol=1e-5)

    def test_initialization_reproducibility(self):
        first, second = initialize(3, seed=2026), initialize(3, seed=2026)
        for name in vars(first):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_invalid_counts_permutations_inputs_and_parameters(self):
        for d, n, h in ((0, 4, None), (3, -1, None), (True, 4, None),
                        (3.5, 4, None), (3, 4, 0), (3, 4, False)):
            with self.subTest(D=d, N=n, H=h), self.assertRaises(ValueError):
                parameter_counts(d, n, h)
        for order in ([], [0, 0], [0, 2], [1, -1], [0., 1.], [[0, 1]], [True, False]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                permutation_matrix(order)
        params = initialize(3)
        for x in ([], [1, 2, 3], np.empty((3, 0)), np.full((3, 4), np.nan)):
            with self.assertRaises(ValueError):
                self_attention(x, params)
        for invalid in (replace(params, w_q=np.zeros((2, 2))),
                        replace(params, b_k=np.zeros((3, 1))),
                        replace(params, w_v=np.full((3, 3), np.inf)),
                        replace(params, b_q=np.full(3, np.nan))):
            with self.assertRaises(ValueError):
                self_attention(np.ones((3, 4)), invalid)
        with self.assertRaises(ValueError):
            initialize(0)


if __name__ == '__main__':
    unittest.main()
