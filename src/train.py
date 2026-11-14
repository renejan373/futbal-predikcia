"""Tréning a ladenie klasifikačných modelov.

Použitie:
    python -m src.train               # natrénuje všetky modely
    python -m src.train --no-tune     # rýchly beh bez ladenia hyperparametrov

Vstup:  data/processed/features.csv
Výstup: models/<model>.pkl, models/metadata.json,
        reports/tables/03_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES_CSV,
    MODEL_METADATA_JSON,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_CLASSES,
    TEST_SEASON,
    TRAIN_SEASONS,
    VALID_SEASON,
)
from src.features import FEATURE_COLUMNS
from src.metrics import compute_metrics

N_SPLITS = 5
N_ITER = 12
SCORING = "neg_log_loss"

MODEL_NAMES_SK = {
    "logistic_regression": "Logistická regresia",
    "random_forest": "Náhodný les",
    "gradient_boosting": "Gradient boosting",
}


def load_data() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Chyba {FEATURES_CSV}. Najprv spusti: python -m src.features"
        )
    features = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    features["season"] = features["season"].astype(str)
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise RuntimeError(f"Chybaju atributy: {missing}")
    return features


def model_zoo() -> dict[str, tuple[Pipeline, dict]]:
    """Vráti slovník modelov a ich prehľadávaných hyperparametrov."""
    return {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__C": loguniform(1e-3, 1e2),
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            {
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_leaf": [2, 5, 10, 20],
                "model__max_features": ["sqrt", 0.5, None],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
        "gradient_boosting": (
            Pipeline(
                steps=[
                    (
                        "model",
                        GradientBoostingClassifier(
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            {
                "model__learning_rate": [0.02, 0.03, 0.05, 0.1],
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.8, 1.0],
                "model__min_samples_leaf": [5, 10, 20],
                "model__max_features": [None, "sqrt"],
            },
        ),
    }


def split_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = features[features["season"].isin(TRAIN_SEASONS)].sort_values("date")
    valid = features[features["season"] == VALID_SEASON].sort_values("date")
    test = features[features["season"] == TEST_SEASON].sort_values("date")
    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError("Niektora z mnozin (train/valid/test) je prazdna.")
    return train, valid, test


def tune_model(
    name: str,
    estimator: Pipeline,
    distributions: dict,
    train: pd.DataFrame,
    tune: bool,
) -> tuple[Pipeline, dict, dict]:
    """Vyladí model na tréningovej sade a vráti najlepší odhad a metriky."""
    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    if not tune:
        fitted = estimator.fit(X_train, y_train)
        return fitted, {}, {"mode": "bez ladenia"}

    search = RandomizedSearchCV(
        estimator,
        param_distributions=distributions,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    info = {
        "mode": "randomized_search",
        "n_iter": N_ITER,
        "cv": f"TimeSeriesSplit({N_SPLITS})",
        "scoring": SCORING,
        "seconds": round(elapsed, 2),
        "best_cv_score": float(search.best_score_),
    }
    print(
        f"    {name}: najlepší CV log loss {-search.best_score_:.4f} "
        f"za {elapsed:.1f} s"
    )
    return search.best_estimator_, dict(search.best_params_), info


def predict_frame(
    model: Pipeline,
    frame: pd.DataFrame,
    model_key: str,
) -> pd.DataFrame:
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS])
    predictions = probabilities.argmax(axis=1)
    result = frame[
        [
            "match_id",
            "season",
            "date",
            "home_team",
            "away_team",
            "ftr",
            "target",
            "odds_h",
            "odds_d",
            "odds_a",
            "implied_h",
            "implied_d",
            "implied_a",
        ]
    ].copy()
    result["model"] = model_key
    result["p_h"] = probabilities[:, 0]
    result["p_d"] = probabilities[:, 1]
    result["p_a"] = probabilities[:, 2]
    result["pred"] = [TARGET_CLASSES[index] for index in predictions]
    result["correct"] = result["pred"] == result["ftr"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Natrenuj modely pre predikciu zapasov")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="preskoci RandomizedSearchCV (rychly test pipeline)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = load_data()
    train, valid, test = split_data(features)
    print(
        f"Data: train={len(train)} ({TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]}), "
        f"valid={len(valid)} ({VALID_SEASON}), test={len(test)} ({TEST_SEASON})"
    )

    predictions: list[pd.DataFrame] = []
    metadata: dict[str, object] = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "features": FEATURE_COLUMNS,
        "target_classes": TARGET_CLASSES,
        "train_seasons": TRAIN_SEASONS,
        "valid_season": VALID_SEASON,
        "test_season": TEST_SEASON,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "models": {},
    }

    for key, (estimator, distributions) in model_zoo().items():
        print(f"\n[{key}] ladenie...")
        best, params, tuning = tune_model(key, estimator, distributions, train, not args.no_tune)

        valid_probabilities = best.predict_proba(valid[FEATURE_COLUMNS])
        valid_metrics = compute_metrics(valid["target"], valid_probabilities)
        print(
            f"    validacia: accuracy={valid_metrics['accuracy']:.3f}, "
            f"log_loss={valid_metrics['log_loss']:.4f}"
        )

        # Finálny model sa dotrénuje na train + valid (test zostáva nedotknutý).
        final = best.fit(
            pd.concat([train, valid])[FEATURE_COLUMNS],
            pd.concat([train, valid])["target"],
        )
        test_probabilities = final.predict_proba(test[FEATURE_COLUMNS])
        test_metrics = compute_metrics(test["target"], test_probabilities)
        print(
            f"    test:     accuracy={test_metrics['accuracy']:.3f}, "
            f"log_loss={test_metrics['log_loss']:.4f}"
        )

        model_path = MODELS_DIR / f"{key}.pkl"
        joblib.dump(final, model_path)
        predictions.append(predict_frame(final, test, key))

        metadata["models"][key] = {  # type: ignore[index]
            "name_sk": MODEL_NAMES_SK[key],
            "file": model_path.name,
            "best_params": params,
            "tuning": tuning,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }

    best_key = min(
        metadata["models"],  # type: ignore[index]
        key=lambda name: metadata["models"][name]["valid_metrics"]["log_loss"],  # type: ignore[index]
    )
    metadata["best_model"] = {
        "key": best_key,
        "name_sk": MODEL_NAMES_SK[best_key],
        "selected_by": f"najnižší log loss na sezóne {VALID_SEASON}",
    }

    predictions_frame = pd.concat(predictions, ignore_index=True)
    predictions_path = TABLES_DIR / "03_test_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8")

    MODEL_METADATA_JSON.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[ok] modely -> {MODELS_DIR}")
    print(f"[ok] metadata -> {MODEL_METADATA_JSON}")
    print(f"[ok] predikcie -> {predictions_path}")
    print(f"[ok] najlepsi model podla validacie: {MODEL_NAMES_SK[best_key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
