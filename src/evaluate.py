"""Vyhodnotenie modelov a porovnanie so stávkovými kurzami.

Vstup:  data/processed/features.csv
        reports/tables/03_test_predictions.csv
        models/metadata.json
        models/random_forest.pkl (pre dôležitosť atribútov)
Výstup: reports/tables/04_* … 09_*
        reports/figures/01_* … 06_*

Použitie:
    python -m src.evaluate
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.config import (
    FEATURES_CSV,
    FIGURES_DIR,
    MODEL_METADATA_JSON,
    MODELS_DIR,
    TABLES_DIR,
    TARGET_CLASSES,
    TARGET_NAMES_SK,
    TEST_SEASON,
)
from src.features import FEATURE_COLUMNS
from src.metrics import compute_metrics, confusion, per_class_metrics

MODEL_ORDER = ["logistic_regression", "random_forest", "gradient_boosting"]
FALLBACK_NAMES = {
    "logistic_regression": "Logistická regresia",
    "random_forest": "Náhodný les",
    "gradient_boosting": "Gradient boosting",
}
MARKET_NAME = "Kurzy Bet365 (favorit)"
BASELINE_HOME = "Vždy výhra domácich"
BASELINE_RANDOM = "Náhodný tip (1/3)"
VALUE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
CLASS_LABELS = {
    0: "H – výhra domácich",
    1: "D – remíza",
    2: "A – výhra hostí",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not MODEL_METADATA_JSON.exists():
        raise FileNotFoundError(
            f"Chyba {MODEL_METADATA_JSON}. Najprv spusti: python -m src.train"
        )
    metadata = json.loads(MODEL_METADATA_JSON.read_text(encoding="utf-8"))

    features = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    features["season"] = features["season"].astype(str)
    test = features[features["season"] == TEST_SEASON].sort_values("date").reset_index(drop=True)

    predictions = pd.read_csv(
        TABLES_DIR / "03_test_predictions.csv", parse_dates=["date"]
    )
    return test, predictions, metadata


def align_predictions(
    test: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    for model_key, group in predictions.groupby("model"):
        aligned = group.set_index("match_id").reindex(test["match_id"])
        if aligned[["p_h", "p_d", "p_a"]].isna().any().any():
            raise RuntimeError(f"Chýbajú predikcie pre model {model_key}")
        probabilities[model_key] = aligned[["p_h", "p_d", "p_a"]].to_numpy()
    return probabilities


def build_comparison_table(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> pd.DataFrame:
    y_true = test["target"].to_numpy()
    market = test[["implied_h", "implied_d", "implied_a"]].to_numpy()

    rows: list[dict[str, object]] = []

    def add(name: str, kind: str, probs: np.ndarray) -> None:
        metrics = compute_metrics(y_true, probs)
        rows.append({"model": name, "typ": kind, **metrics})

    add(BASELINE_HOME, "baseline", np.tile([1.0, 0.0, 0.0], (len(test), 1)))
    add(BASELINE_RANDOM, "baseline", np.tile([1 / 3, 1 / 3, 1 / 3], (len(test), 1)))
    add(MARKET_NAME, "trh", market)

    names = {
        key: metadata.get("models", {}).get(key, {}).get("name_sk", FALLBACK_NAMES.get(key, key))
        for key in MODEL_ORDER
    }
    for key in MODEL_ORDER:
        add(names[key], "model", probabilities[key])

    table = pd.DataFrame(rows)
    order = {
        BASELINE_HOME: 0,
        BASELINE_RANDOM: 1,
        MARKET_NAME: 2,
        names["logistic_regression"]: 3,
        names["random_forest"]: 4,
        names["gradient_boosting"]: 5,
    }
    table["_order"] = table["model"].map(order).fillna(99)
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return table


def build_per_class_table(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> pd.DataFrame:
    y_true = test["target"].to_numpy()
    rows = []
    market = test[["implied_h", "implied_d", "implied_a"]].to_numpy()
    sources = {MARKET_NAME: market}
    for key in MODEL_ORDER:
        name = metadata.get("models", {}).get(key, {}).get("name_sk", FALLBACK_NAMES[key])
        sources[name] = probabilities[key]

    for name, probs in sources.items():
        metrics = per_class_metrics(y_true, probs)
        metrics["model"] = name
        rows.append(metrics)
    table = pd.concat(rows, ignore_index=True)
    table["class"] = table["class_id"].map(CLASS_LABELS)
    return table[["model", "class", "precision", "recall", "f1", "support"]]


def build_confusion_table(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> pd.DataFrame:
    y_true = test["target"].to_numpy()
    market = test[["implied_h", "implied_d", "implied_a"]].to_numpy()
    sources = {MARKET_NAME: market}
    for key in MODEL_ORDER:
        name = metadata.get("models", {}).get(key, {}).get("name_sk", FALLBACK_NAMES[key])
        sources[name] = probabilities[key]

    rows = []
    for name, probs in sources.items():
        matrix = confusion(y_true, probs)
        for actual_index, actual in enumerate(TARGET_CLASSES):
            for predicted_index, predicted in enumerate(TARGET_CLASSES):
                rows.append(
                    {
                        "model": name,
                        "skutocnost": actual,
                        "predikcia": predicted,
                        "pocet": int(matrix[actual_index, predicted_index]),
                    }
                )
    return pd.DataFrame(rows)


def build_odds_comparison(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> pd.DataFrame:
    y_true = test["target"].to_numpy()
    market = test[["implied_h", "implied_d", "implied_a"]].to_numpy()
    market_metrics = compute_metrics(y_true, market)
    market_favorite = market.argmax(axis=1)

    rows = []
    for key in MODEL_ORDER:
        name = metadata.get("models", {}).get(key, {}).get("name_sk", FALLBACK_NAMES[key])
        probs = probabilities[key]
        metrics = compute_metrics(y_true, probs)
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "delta_log_loss_vs_trh": metrics["log_loss"] - market_metrics["log_loss"],
                "zhoda_s_favoritom_trhu": float(
                    np.mean(probs.argmax(axis=1) == market_favorite)
                ),
                "priemerna_istota": float(probs.max(axis=1).mean()),
            }
        )
    rows.append(
        {
            "model": MARKET_NAME,
            "accuracy": market_metrics["accuracy"],
            "log_loss": market_metrics["log_loss"],
            "brier": market_metrics["brier"],
            "delta_log_loss_vs_trh": 0.0,
            "zhoda_s_favoritom_trhu": 1.0,
            "priemerna_istota": float(market.max(axis=1).mean()),
        }
    )
    return pd.DataFrame(rows)


def _bootstrap_roi(profits: np.ndarray, iterations: int = 2000) -> tuple[float, float]:
    """95 % interval spoľahlivosti pre ROI (bootstrap)."""
    if len(profits) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    samples = rng.choice(profits, size=(iterations, len(profits)), replace=True)
    rois = samples.mean(axis=1)
    low, high = np.percentile(rois, [2.5, 97.5])
    return float(low), float(high)


def _summarise_strategy(
    name: str,
    threshold: float,
    profits: np.ndarray,
) -> dict[str, object]:
    n_bets = len(profits)
    total = float(profits.sum())
    roi = total / n_bets if n_bets else np.nan
    low, high = _bootstrap_roi(profits)
    return {
        "model": name,
        "hranica": threshold,
        "pocet_stavok": n_bets,
        "vyhry": int((profits > 0).sum()),
        "uspesnost": float((profits > 0).mean()) if n_bets else np.nan,
        "vsadene": float(n_bets),
        "zisk": total,
        "roi": roi,
        "roi_ci_low": low,
        "roi_ci_high": high,
    }


def build_value_bets(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> pd.DataFrame:
    odds = test[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    y_true = test["target"].to_numpy()
    market = test[["implied_h", "implied_d", "implied_a"]].to_numpy()
    rows: list[dict[str, object]] = []

    for key in MODEL_ORDER:
        name = metadata.get("models", {}).get(key, {}).get("name_sk", FALLBACK_NAMES[key])
        probs = probabilities[key]
        edges = probs - market
        for threshold in VALUE_THRESHOLDS:
            bets = edges > threshold
            profits = []
            for match_index, outcome in zip(*np.nonzero(bets)):
                if y_true[match_index] == outcome:
                    profits.append(odds[match_index, outcome] - 1.0)
                else:
                    profits.append(-1.0)
            rows.append(
                _summarise_strategy(name, threshold, np.asarray(profits, dtype=float))
            )

    # Referenčné stratégie: slepé stávkovanie na jeden výsledok.
    for outcome_index, label in enumerate(("Vždy výhra domácich", "Vždy remíza", "Vždy výhra hostí")):
        profits = np.where(
            y_true == outcome_index,
            odds[:, outcome_index] - 1.0,
            -1.0,
        )
        rows.append(_summarise_strategy(label, np.nan, profits))

    return pd.DataFrame(rows)


def build_confidence_table(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> pd.DataFrame:
    y_true = test["target"].to_numpy()
    bins = [0.0, 0.4, 0.5, 0.6, 1.0]
    labels = ["< 40 %", "40–50 %", "50–60 %", "≥ 60 %"]
    rows = []
    for key in MODEL_ORDER:
        name = metadata.get("models", {}).get(key, {}).get("name_sk", FALLBACK_NAMES[key])
        probs = probabilities[key]
        confidence = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        groups = pd.cut(confidence, bins=bins, labels=labels, right=False)
        for label in labels:
            mask = groups == label
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "model": name,
                    "pasmo_istoty": label,
                    "pocet": int(mask.sum()),
                    "uspesnost": float(np.mean(predictions[mask] == y_true[mask])),
                }
            )
    return pd.DataFrame(rows)


def plot_class_distribution(features: pd.DataFrame) -> None:
    counts = features["ftr"].value_counts().reindex(TARGET_CLASSES)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Výhra domácich", "Remíza", "Výhra hostí"],
        counts.values,
        color=["#2a9d8f", "#e9c46a", "#264653"],
    )
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{count}\n({count / counts.sum():.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Počet zápasov")
    ax.set_title("Rozdelenie výsledkov (2021/22 – 2025/26)")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=300)
    plt.close(fig)


def plot_goals_distribution(features: pd.DataFrame) -> None:
    total_goals = features["fthg"] + features["ftag"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.5, total_goals.max() + 1.5, 1)
    ax.hist(total_goals, bins=bins, color="#457b9d", edgecolor="white")
    ax.axvline(total_goals.mean(), color="#e63946", linestyle="--", label=f"Priemer {total_goals.mean():.2f}")
    ax.set_xlabel("Počet gólov v zápase")
    ax.set_ylabel("Počet zápasov")
    ax.set_title("Rozdelenie počtu gólov")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_goals_distribution.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrices(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    y_true = test["target"].to_numpy()
    market = test[["implied_h", "implied_d", "implied_a"]].to_numpy()
    sources = []
    for key in MODEL_ORDER:
        name = metadata.get("models", {}).get(key, {}).get("name_sk", FALLBACK_NAMES[key])
        sources.append((name, probabilities[key]))
    sources.append((MARKET_NAME, market))

    fig, axes = plt.subplots(1, len(sources), figsize=(4 * len(sources), 3.8))
    for ax, (name, probs) in zip(axes, sources):
        matrix = confusion(y_true, probs)
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                color = "white" if value > matrix.max() / 2 else "black"
                ax.text(column, row, str(value), ha="center", va="center", color=color)
        ax.set_xticks(range(3), TARGET_CLASSES)
        ax.set_yticks(range(3), TARGET_CLASSES)
        ax.set_xlabel("Predikcia")
        if ax is axes[0]:
            ax.set_ylabel("Skutočnosť")
        ax.set_title(name, fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.7, label="Počet zápasov")
    fig.suptitle(f"Matice zámen – testovacia sezóna {TEST_SEASON}")
    fig.savefig(FIGURES_DIR / "03_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    y_true = test["target"].to_numpy()
    best_key = metadata.get("best_model", {}).get("key", "random_forest")
    if best_key not in probabilities:
        best_key = MODEL_ORDER[0]
    best_name = metadata.get("models", {}).get(best_key, {}).get(
        "name_sk", FALLBACK_NAMES.get(best_key, best_key)
    )
    market = test[["implied_h", "implied_d", "implied_a"]].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for class_index, (ax, label) in enumerate(zip(axes, TARGET_CLASSES)):
        binary = (y_true == class_index).astype(int)
        for probs, name, color in (
            (probabilities[best_key], best_name, "#1d3557"),
            (market, "Kurzy Bet365", "#e63946"),
        ):
            fraction, mean_predicted = calibration_curve(
                binary, probs[:, class_index], n_bins=8, strategy="quantile"
            )
            ax.plot(mean_predicted, fraction, marker="o", label=name, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideálna kalibrácia")
        ax.set_title(f"Trieda {label}")
        ax.set_xlabel("Predikovaná pravdepodobnosť")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Skutočná frekvencia")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Kalibrácia pravdepodobností – {best_name} vs. trh ({TEST_SEASON})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_calibration.png", dpi=300)
    plt.close(fig)


def plot_feature_importance() -> None:
    path = MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        return
    model = joblib.load(path)
    importances = model.named_steps["model"].feature_importances_
    order = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [FEATURE_COLUMNS[index] for index in order],
        importances[order],
        color="#2a9d8f",
    )
    ax.set_xlabel("Relatívna dôležitosť")
    ax.set_title("15 najdôležitejších atribútov (Random Forest)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_feature_importance.png", dpi=300)
    plt.close(fig)


def plot_value_bets(value_bets: pd.DataFrame) -> None:
    models = value_bets[value_bets["hranica"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    thresholds = sorted(models["hranica"].dropna().unique())
    names = list(models["model"].unique())
    for position, name in enumerate(names):
        subset = models[models["model"] == name].set_index("hranica").reindex(thresholds)
        ax.bar(
            np.arange(len(thresholds)) + position * width,
            subset["roi"].fillna(0) * 100,
            width,
            label=name,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(thresholds)) + width, [f"{t:.0%}" for t in thresholds])
    ax.set_xlabel("Minimálna výhoda oproti trhu")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Simulácia value bets – návratnosť vkladov")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_value_bets_roi.png", dpi=300)
    plt.close(fig)


def dataframe_to_markdown(table: pd.DataFrame, floatfmt: str = ".3f") -> str:
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(":---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    comparison: pd.DataFrame,
    odds: pd.DataFrame,
    value_bets: pd.DataFrame,
    metadata: dict,
) -> None:
    best_key = metadata.get("best_model", {}).get("key", "random_forest")
    best_name = metadata.get("models", {}).get(best_key, {}).get(
        "name_sk", FALLBACK_NAMES.get(best_key, best_key)
    )
    model_rows = comparison[comparison["typ"] == "model"]
    best_accuracy = model_rows.sort_values("accuracy", ascending=False).iloc[0]
    market_row = comparison[comparison["typ"] == "trh"].iloc[0]

    lines = [
        f"# Výsledky – testovacia sezóna {TEST_SEASON}",
        "",
        "## Porovnanie modelov",
        "",
        dataframe_to_markdown(
            comparison[["model", "typ", "accuracy", "balanced_accuracy", "f1_macro", "log_loss", "brier"]]
        ),
        "",
        f"- Najvyššiu presnosť dosiahol **{best_accuracy['model']}** "
        f"({best_accuracy['accuracy']:.1%}).",
        f"- Kurzový favorit stávkových kancelárií mal presnosť "
        f"{market_row['accuracy']:.1%} a log loss {market_row['log_loss']:.4f}.",
        f"- Model vybraný podľa validačnej sezóny: **{best_name}**.",
        "",
        "## Porovnanie s kurzami",
        "",
        dataframe_to_markdown(odds),
        "",
        "## Simulácia value bets",
        "",
        dataframe_to_markdown(value_bets),
        "",
        "Hodnoty `roi` sú návratnosť pri rovnom vklade 1 jednotky na každú stávku. "
        "`roi_ci_low` a `roi_ci_high` tvoria 95 % bootstrapový interval spoľahlivosti.",
        "",
        "> **Pozor pri interpretácii:** ide o jedinú testovaciu sezónu, takže interval "
        "spoľahlivosti je široký a kladné ROI samo o sebe neznamená skutočnú výhodu "
        "oproti trhu. Referenčné riadky „Vždy …“ ukazujú, aké ROI by prinieslo slepé "
        "stávkovanie na jeden výsledok – v sezóne 2025/26 bolo napríklad stávkovanie "
        "na remízy nezvyčajne úspešné.",
    ]
    (TABLES_DIR / "04_vysledky.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    features["season"] = features["season"].astype(str)
    test, predictions, metadata = load_inputs()
    probabilities = align_predictions(test, predictions)

    comparison = build_comparison_table(test, probabilities, metadata)
    per_class = build_per_class_table(test, probabilities, metadata)
    confusion_table = build_confusion_table(test, probabilities, metadata)
    odds = build_odds_comparison(test, probabilities, metadata)
    value_bets = build_value_bets(test, probabilities, metadata)
    confidence = build_confidence_table(test, probabilities, metadata)

    comparison.to_csv(TABLES_DIR / "04_model_comparison.csv", index=False)
    per_class.to_csv(TABLES_DIR / "05_per_class_metrics.csv", index=False)
    confusion_table.to_csv(TABLES_DIR / "06_confusion_matrices.csv", index=False)
    odds.to_csv(TABLES_DIR / "07_odds_comparison.csv", index=False)
    value_bets.to_csv(TABLES_DIR / "08_value_bets.csv", index=False)
    confidence.to_csv(TABLES_DIR / "09_accuracy_by_confidence.csv", index=False)
    write_summary(comparison, odds, value_bets, metadata)

    plot_class_distribution(features)
    plot_goals_distribution(features)
    plot_confusion_matrices(test, probabilities, metadata)
    plot_calibration(test, probabilities, metadata)
    plot_feature_importance()
    plot_value_bets(value_bets)

    print(comparison[["model", "accuracy", "log_loss", "brier"]].to_string(index=False))
    print(f"\n[ok] tabulky -> {TABLES_DIR}")
    print(f"[ok] grafy -> {FIGURES_DIR}")
    print(f"[ok] zhrnutie -> {TABLES_DIR / '04_vysledky.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
