import importlib.metadata
import json
from itertools import permutations
from pathlib import Path
import platform

import numpy as np
import pandas as pd

try:
    from .attention import (
        parameter_counts,
        initialize,
        self_attention,
        permutation_matrix,
        stable_softmax,
        softmax_jacobian,
    )
except ImportError:
    from attention import (
        parameter_counts,
        initialize,
        self_attention,
        permutation_matrix,
        stable_softmax,
        softmax_jacobian,
    )


LOGITS = np.array([-3.0, 1.0, 100.0, 5.0, -1.0])
SEED = 2026
RTOL = 1e-5
ATOL = 1e-7


def run_experiments(lab_dir=None):
    lab_dir = Path(lab_dir) if lab_dir is not None else Path(__file__).resolve().parent
    results = lab_dir / "results"
    results.mkdir(exist_ok=True)
    tables = {}
    tables["parameter_counts"] = pd.DataFrame(
        [parameter_counts(d, n) for d in (2, 8, 32) for n in (2, 4, 8, 16, 32, 64, 128)]
    )
    x = np.array([[0.2, 1.0, -0.5, 0.7], [1.2, -0.3, 0.8, 0.1], [-0.4, 0.6, 1.1, -0.2]])
    parameters = initialize(3, SEED)
    p = permutation_matrix([1, 3, 0, 2])
    original = self_attention(x, parameters)
    permuted = self_attention(x @ p, parameters)
    for name, array in {
        "input": x,
        "permutation": p,
        "values": original.values,
        "keys": original.keys,
        "queries": original.queries,
        "attention": original.weights,
        "attention_permuted": permuted.weights,
        "output": original.output,
        "output_permuted": permuted.output,
    }.items():
        tables[name] = pd.DataFrame(array)
    np.savez_compressed(results / "attention_parameters.npz", **vars(parameters))
    rows = []
    rng = np.random.default_rng(SEED)
    for d, n in ((2, 3), (3, 4), (8, 12)):
        inputs = x if (d, n) == (3, 4) else rng.normal(size=(d, n))
        params = initialize(d, SEED)
        orders = (
            list(permutations(range(n)))
            if n <= 4
            else [rng.permutation(n) for _ in range(30)]
        )
        for scaled in (False, True):
            base = self_attention(inputs, params, scaled=scaled)
            for number, order in enumerate(orders):
                matrix = permutation_matrix(order)
                actual = self_attention(inputs @ matrix, params, scaled=scaled)
                error = np.max(np.abs(actual.output - base.output @ matrix))
                rows.append(
                    {
                        "D": d,
                        "N": n,
                        "scaled": scaled,
                        "permutation_id": number,
                        "output_max_error": error,
                        "attention_max_error": np.max(
                            np.abs(actual.weights - matrix.T @ base.weights @ matrix)
                        ),
                    }
                )
    tables["equivariance"] = pd.DataFrame(rows)
    probabilities = stable_softmax(LOGITS)
    jacobian = softmax_jacobian(LOGITS)
    tables["softmax"] = pd.DataFrame(
        {"i": np.arange(1, 6), "z": LOGITS, "y": probabilities}
    )
    tables["jacobian"] = pd.DataFrame(
        jacobian,
        index=[f"y{i}" for i in range(1, 6)],
        columns=[f"z{i}" for i in range(1, 6)],
    )
    for name, table in tables.items():
        table.to_csv(
            results / f"{name}.csv", index=name == "jacobian", encoding="utf-8-sig", float_format="%.4e"
        )
    settings = {
        "seed": SEED,
        "logits": LOGITS.tolist(),
        "X_shape_convention": "D x N",
        "softmax_axis_attention": 0,
        "MLP_hidden_assumption": "H = D*N for comparison table",
        "significant_digits": 5,
        "relative_tolerance": RTOL,
        "absolute_tolerance": ATOL,
        "permutation_checks": len(rows),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "matplotlib", "nbformat", "nbclient")
        },
    }
    (results / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print("Число проверок перестановок:", len(rows))
    print(
        "Максимальное расхождение выходов:",
        f'{tables["equivariance"].output_max_error.max():.4e}',
    )
    print("J33:", f"{jacobian[2, 2]:.4e}")
    return tables


if __name__ == "__main__":
    run_experiments()
