"""Воспроизводимые эксперименты: python Lab5/experiments.py [--epochs 3000]"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

try:
    from .optimization import (
        PARAMETERS,
        WEIGHT_MASK,
        initialize,
        objective,
        predict,
        save_model,
        train,
    )
except ImportError:
    from optimization import (
        PARAMETERS,
        WEIGHT_MASK,
        initialize,
        objective,
        predict,
        save_model,
        train,
    )


MODELS = ("linear", "quadratic", "relu")
METHODS = ("gd", "sgd", "nesterov", "adam")
BATCH_SIZES = (1, 3, 6, 12)
SEEDS = (2026, 2027, 2028)
PENALTIES = (0.0, 0.1, 1.0)
HISTORY_COLUMNS = (
    "epoch",
    "updates",
    "SSE",
    "objective",
    "gradient_inf",
    "weight_norm",
    "learning_rate",
    "derivative_sample_evaluations",
)
# Различие шагов фиксируется до сравнения: Adam нормирует компоненты градиента
LEARNING_RATES = {
    "linear": {"gd": 0.02, "sgd": 0.005, "nesterov": 0.01, "adam": 0.02},
    "quadratic": {"gd": 0.01, "sgd": 0.0025, "nesterov": 0.01, "adam": 0.02},
    "relu": {"gd": 0.01, "sgd": 0.005, "nesterov": 0.01, "adam": 0.02},
}


def load_data(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("data.csv")
    frame = pd.read_csv(path)
    if (
        list(frame.columns) != ["x", "y"]
        or frame.empty
        or frame.isna().any().any()
        or not np.isfinite(frame.to_numpy(float)).all()
    ):
        raise ValueError("Нужна непустая конечная таблица x,y без пропусков")
    return frame


def reference_solution(x, y, model, l2=0.0):
    """Независимый контроль для двух выпуклых моделей; при обучении не используется"""
    if model == "linear":
        a = np.column_stack([x, np.ones(len(x))])
    elif model == "quadratic":
        a = np.column_stack([x**2, x, np.ones(len(x))])
    else:
        raise ValueError("Аналитический контроль предусмотрен только для полиномов")
    if l2 == 0:
        theta = np.linalg.lstsq(a, y, rcond=None)[0]
    else:
        theta = np.linalg.solve(a.T @ a + l2 * np.diag(WEIGHT_MASK[model]), a.T @ y)
    return theta


def run_experiments(lab_dir=None, *, max_epochs=3000, seeds=SEEDS, penalties=PENALTIES):
    lab_dir = Path(lab_dir) if lab_dir is not None else Path(__file__).resolve().parent
    result_dir, model_dir = lab_dir / "results", lab_dir / "models"
    result_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    frame = load_data(lab_dir / "data.csv")
    x, y = frame.x.to_numpy(float), frame.y.to_numpy(float)
    if len(x) != 12:
        raise ValueError("Конфигурация эксперимента рассчитана на 12 точек задания")
    settings = {
        "max_epochs": max_epochs,
        "seeds": list(seeds),
        "penalties": list(penalties),
        "models": MODELS,
        "methods": METHODS,
        "batch_sizes": BATCH_SIZES,
        "learning_rates": LEARNING_RATES,
        "tolerance": 1e-5,
        "relative_step": 1e-5,
        "decay_epochs": 300.0,
        "beta": 0.9,
        "gamma": 0.999,
        "epsilon_adam": 1e-8,
        "objective": "SSE + lambda * sum(weights**2), biases excluded",
        "batch_objective": "n / len(batch) * SSE_batch + lambda * sum(weights**2)",
        "schedule": "constant for GD; lr0 / (1 + epoch / 300) otherwise",
        "initialization": "polynomials uniform(-0.1,0.1); ReLU [1,-.35,-1,1.3,.25,.25,.5] + uniform(-.04,.04)",
        "stopping": "full central-difference gradient infinity norm < 1e-5, checked every epoch",
        "history_every": 20,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "pandas", "matplotlib", "nbformat", "nbclient")
        },
    }
    (result_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows, histories, parameters, predictions, initial_rows, reference_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for model in MODELS:
        for seed in seeds:
            for name, value in zip(PARAMETERS[model], initialize(model, seed)):
                initial_rows.append(
                    {"model": model, "seed": seed, "parameter": name, "value": value}
                )
        for l2 in penalties:
            if model != "relu":
                ref = reference_solution(x, y, model, l2)
                reference_rows.append(
                    {
                        "model": model,
                        "l2": l2,
                        "SSE": float(objective(ref, x, y, model)),
                        "objective": float(objective(ref, x, y, model, l2=l2)),
                        "theta": json.dumps(ref.tolist()),
                    }
                )
            for seed in seeds:
                for method in METHODS:
                    for batch in (12,) if method == "gd" else BATCH_SIZES:
                        run_id = f"{model}_{method}_b{batch}_l{l2:g}_s{seed}"
                        result = train(
                            x,
                            y,
                            model,
                            method=method,
                            batch_size=batch,
                            learning_rate=LEARNING_RATES[model][method],
                            max_epochs=max_epochs,
                            l2=l2,
                            seed=seed,
                        )
                        prediction = predict(x, result.theta, model)
                        row = {
                            "run_id": run_id,
                            "model": model,
                            "method": method,
                            "batch_size": batch,
                            "l2": l2,
                            "seed": seed,
                            "learning_rate": LEARNING_RATES[model][method],
                            "epochs": result.epochs,
                            "updates": result.updates,
                            "converged": result.converged,
                            "SSE": result.sse,
                            "RMSE": np.sqrt(result.sse / len(x)),
                            "R2": 1 - result.sse / np.sum((y - y.mean()) ** 2),
                            "objective": result.loss,
                            "gradient_inf": result.grad_norm,
                            "weight_norm": result.weight_norm,
                            "gradient_calls": result.gradient_calls,
                            "derivative_sample_evaluations": result.derivative_sample_evaluations,
                        }
                        rows.append(row)
                        hist = pd.DataFrame(result.history, columns=HISTORY_COLUMNS)
                        hist.insert(0, "run_id", run_id)
                        histories.append(hist)
                        for name, value in zip(PARAMETERS[model], result.theta):
                            parameters.append(
                                {"run_id": run_id, "parameter": name, "value": value}
                            )
                        for i, value in enumerate(prediction):
                            predictions.append(
                                {
                                    "run_id": run_id,
                                    "point": i + 1,
                                    "x": x[i],
                                    "y": y[i],
                                    "prediction": value,
                                    "residual": y[i] - value,
                                }
                            )
                        if seed == seeds[0] and method == "adam" and batch == 3:
                            save_model(
                                model_dir / f"{model}_adam_b3_l{l2:g}.npz",
                                model,
                                result,
                            )
                print(
                    f"{model}, lambda={l2:g}, seed={seed}: завершено {len(rows)} запусков",
                    flush=True,
                )
    training, history, params = (
        pd.DataFrame(rows),
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(parameters),
    )
    tables = {
        "training": training,
        "history": history,
        "parameters": params,
        "predictions": pd.DataFrame(predictions),
        "initial_parameters": pd.DataFrame(initial_rows),
        "reference": pd.DataFrame(reference_rows),
    }
    summary = training.groupby(
        ["model", "method", "batch_size", "l2"], as_index=False
    ).agg(
        runs=("seed", "size"),
        SSE_mean=("SSE", "mean"),
        SSE_std=("SSE", "std"),
        SSE_min=("SSE", "min"),
        SSE_max=("SSE", "max"),
        RMSE_mean=("RMSE", "mean"),
        objective_mean=("objective", "mean"),
        weight_norm_mean=("weight_norm", "mean"),
        gradient_mean=("gradient_inf", "mean"),
        converged_runs=("converged", "sum"),
        updates_mean=("updates", "mean"),
    )
    tables["summary"] = summary
    for name, table in tables.items():
        table.to_csv(result_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    return training, history, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3000)
    args = parser.parse_args()
    run_experiments(max_epochs=args.epochs)
