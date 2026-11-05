import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import (golden_section, target_function, numerical_gradient, predict,
                          objective, initialize, minibatches, optimizer_step, train,
                          save_model, predict_saved)
from experiments import load_data, reference_solution


class OptimizationTests(unittest.TestCase):
    def test_golden_section_known_minimum_and_endpoints(self):
        result = golden_section(lambda z: (z - .37) ** 2 + 2, -2, 4)
        self.assertAlmostEqual(result.x, .37, places=7)
        self.assertLessEqual(result.right - result.left, 1e-10)
        self.assertAlmostEqual(result.fun, 2)
        boundary = golden_section(lambda z: z, 0, 1)
        self.assertLess(boundary.x, 1e-9)
        with self.assertRaises(RuntimeError):
            golden_section(lambda z: z*z, -1, 1, max_iter=1)

    def test_assignment_minimum_against_independent_derivative(self):
        from scipy.optimize import brentq
        derivative = lambda z: (10 * (z - .65) * np.exp(-10 * (z - .65) ** 2)
                                 + 45 * (z - .35) * np.exp(-50 * (z - .35) ** 2))
        stationary = brentq(derivative, .35, .45, xtol=1e-14)
        result = golden_section(target_function, 0, 1)
        self.assertAlmostEqual(result.x, stationary, places=7)
        self.assertAlmostEqual(result.fun, .32860365746917, places=12)

    def test_central_difference_with_known_gradient(self):
        theta = np.array([.2, -3., 10.])
        function = lambda p: p[:, 0] ** 3 + 2*p[:, 1]**2 - 4*p[:, 2]
        np.testing.assert_allclose(numerical_gradient(function, theta), [.12, -12, -4], atol=1e-8)

    def test_all_three_models_and_output_relu(self):
        x = np.array([-1., 0., 2.])
        np.testing.assert_allclose(predict(x, [2., 3.], "linear"), [1, 3, 7])
        np.testing.assert_allclose(predict(x, [2., -1., 3.], "quadratic"), [6, 3, 9])
        theta = np.array([1., 0., -1., 0., 2., -1., -.5])
        np.testing.assert_allclose(predict(x, theta, "relu"), [0, 0, 3.5])
        np.testing.assert_allclose(predict(x, np.stack([theta, theta]), "relu"), [[0, 0, 3.5]]*2)

    def test_l2_excludes_bias_and_batch_gradient_is_unbiased(self):
        x, y, theta = np.array([0., 1., 2., 3., 4.]), np.arange(5.) + 1, np.array([.4, .2])
        self.assertAlmostEqual(float(objective([2., 100.], [], [], "linear", l2=.5)), 2.)
        full = numerical_gradient(lambda t: objective(t, x, y, "linear", l2=.3), theta)
        parts = [numerical_gradient(lambda t: objective(t, x[i:i+1], y[i:i+1], "linear",
                                                       l2=.3, scale=len(x)), theta) for i in range(len(x))]
        np.testing.assert_allclose(np.mean(parts, axis=0), full, atol=1e-8)

    def test_batches_cover_epoch_once_including_short_last_batch(self):
        batches = minibatches(11, 4, np.random.default_rng(2026))
        self.assertEqual([len(b) for b in batches], [4, 4, 3])
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(11))

    def test_gd_one_step_and_nesterov_lookahead_from_lecture(self):
        x, y, initial = np.array([0., 1.]), np.array([1., 3.]), np.array([.5, .2])
        # Известный градиент SSE для y_hat = .5*x+.2: [-4.6, -6.2].
        result = train(x, y, "linear", initial=initial, learning_rate=.01, max_epochs=1)
        np.testing.assert_allclose(result.theta, [.546, .262], atol=1e-10)
        # Два шага Нестерова с beta=.9 и множителем (1-beta) из лекции.
        theta, momentum = initial.copy(), np.zeros(2)
        for _ in range(2):
            lookahead = theta - .01 * .9 * momentum
            residual = lookahead[0] * x + lookahead[1] - y
            gradient = np.array([2 * residual @ x, 2 * residual.sum()])
            momentum = .9 * momentum + .1 * gradient
            theta -= .01 * momentum
        trained = train(x, y, "linear", method="nesterov", initial=initial,
                        learning_rate=.01, max_epochs=2, decay=1e20)
        np.testing.assert_allclose(trained.theta, theta, atol=1e-10)

    def test_adam_bias_correction_two_known_steps(self):
        theta, first, second = np.array([1.]), np.array([0.]), np.array([0.])
        theta, first, second = optimizer_step(theta, np.array([2.]), first, second, 1,
                                              method="adam", learning_rate=.1)
        self.assertAlmostEqual(theta[0], .9, places=8)
        theta, first, second = optimizer_step(theta, np.array([0.]), first, second, 2,
                                              method="adam", learning_rate=.1)
        expected = .9 - .1 * (.18 / .19) / (np.sqrt(.003996 / .001999) + 1e-8)
        self.assertAlmostEqual(theta[0], expected, places=8)

    def test_learning_against_ols_and_ridge(self):
        frame = load_data()
        for l2 in (0., 1.):
            reference = reference_solution(frame.x.to_numpy(), frame.y.to_numpy(), "quadratic", l2)
            fitted = train(frame.x, frame.y, "quadratic", learning_rate=.01, max_epochs=5000, l2=l2)
            self.assertTrue(fitted.converged)
            np.testing.assert_allclose(fitted.theta, reference, atol=3e-5)

    def test_stopping_reproducibility_and_saved_weights(self):
        x = np.linspace(0, 1, 6)
        result = train(x, 2*x + 3, "linear", initial=[2., 3.])
        self.assertTrue(result.converged)
        self.assertEqual(result.updates, 0)
        limited = train(x, 2*x + 3, "linear", max_epochs=0)
        self.assertFalse(limited.converged)
        kwargs = dict(method="adam", batch_size=2, max_epochs=15)
        result = train(x, x*x, "relu", **kwargs)
        repeated = train(x, x*x, "relu", **kwargs)
        np.testing.assert_array_equal(result.theta, repeated.theta)
        self.assertEqual(result.updates, 45)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            save_model(path, "relu", result)
            np.testing.assert_array_equal(predict_saved(path, x), predict(x, result.theta, "relu"))

    def test_reject_invalid_data_and_hyperparameters(self):
        for kwargs in ({"learning_rate": 0}, {"l2": -1}, {"batch_size": 0},
                       {"batch_size": 2}, {"max_epochs": -1}, {"beta": 1},
                       {"relative_step": np.nan}, {"initial": [1.]}, {"method": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train(np.arange(3.), np.ones(3), "linear", **kwargs)
        with self.assertRaises(ValueError):
            train([1., np.nan], [1., 2.], "linear")
        with self.assertRaises(ValueError):
            golden_section(target_function, 1, 0)


if __name__ == "__main__":
    unittest.main()
