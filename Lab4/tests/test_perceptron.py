import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perceptron import initialize, predict, loss_gradient, train, save_model, predict_saved


class PerceptronTests(unittest.TestCase):
    def test_gradients_on_known_example(self):
        x = np.array([[1., 2.], [-1., 1.]])
        y = np.array([1., -1.])
        params = [np.array([[1., -1.], [.5, 1.]]), np.zeros(2),
                  np.array([2., -1.]), np.array([.5])]
        np.testing.assert_allclose(predict(x, params), [3.5, -1.5])
        loss, gradients = loss_gradient(x, y, params)
        self.assertAlmostEqual(loss, 6.5)
        expected = [np.array([[10., -6.], [20., -9.]]), np.array([10., -4.]),
                    np.array([10., 3.]), np.array([4.])]
        for actual, reference in zip(gradients, expected):
            np.testing.assert_allclose(actual, reference)

    def test_heaviside_at_zero(self):
        params = [np.zeros((1, 1)), np.zeros(1), np.array([2.]), np.zeros(1)]
        _, gradients = loss_gradient(np.ones((1, 1)), np.ones(1), params)
        np.testing.assert_allclose(gradients[0], [[-4.]])
        np.testing.assert_allclose(gradients[1], [-4.])

    def test_one_step_and_decay(self):
        x, y = np.ones((4, 2)), np.arange(4.)
        initial = initialize(2, 3)
        _, gradients = loss_gradient(x, y, initial)
        result = train(x, y, 3, learning_rate=.001, max_iter=1,
                       tolerance=1e-15, initial=initial)
        for actual, parameter, gradient in zip(result.params, initial, gradients):
            np.testing.assert_allclose(actual, parameter - .001 * gradient)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        decay_result = train(x, y, 3, learning_rate=.001, max_iter=1,
                             tolerance=1e-15, initial=initial, schedule='decay')
        self.assertAlmostEqual(decay_result.history[-1, 3], .001 / (1 + 1 / 1000))
        for actual, reference in zip(result.params, decay_result.params):
            np.testing.assert_array_equal(actual, reference)

    def test_convergence_at_known_solution(self):
        x = np.arange(1., 5.)[:, None]
        initial = [np.ones((1, 1)), np.zeros(1), np.array([2.]), np.array([3.])]
        result = train(x, 2 * x[:, 0] + 3, 1, initial=initial)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.loss, 0)

    def test_learning_reproducibility_and_saved_weights(self):
        x = np.linspace(0, 1, 40)[:, None]
        y = 2 * x[:, 0] + .3
        original = x.copy()
        initial = [np.array([[.5, .2]]), np.zeros(2), np.array([.3, .4]), np.zeros(1)]
        result = train(x, y, 2, learning_rate=.001, tolerance=1e-4,
                       max_iter=10000, initial=initial)
        self.assertLess(result.loss, 1e-5)
        np.testing.assert_array_equal(x, original)
        repeated = train(x, y, 2, max_iter=2)
        np.testing.assert_array_equal(predict(x, repeated.params),
                                      predict(x, train(x, y, 2, max_iter=2).params))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            save_model(path, result)
            np.testing.assert_allclose(predict_saved(path, x), predict(x, result.params))

    def test_invalid_parameters(self):
        for kwargs in ({'learning_rate': 0}, {'tolerance': -1}, {'max_iter': -1},
                       {'schedule': 'wrong'}, {'learning_rate': np.nan}):
            with self.assertRaises(ValueError):
                train(np.ones((3, 2)), np.ones(3), 2, **kwargs)


if __name__ == '__main__':
    unittest.main()
