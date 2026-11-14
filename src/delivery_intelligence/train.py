
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.delivery_intelligence.features import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42


def create_preprocessor():
    """Cria as transformações para cada tipo de variável."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """Calcula as métricas do modelo."""

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
    }

    print(f"\n--- {name} ---")

    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return metrics



def train():
    """Treina, ajusta e avalia os modelos."""

    print("Carregando dados...")

    X, y = prepare_features()

    # 60% treinamento, 40% restante
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Divide os 40% restantes em validação e teste
    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
    )

    print("\n--- Divisão dos dados ---")
    print(f"Treinamento: {len(X_train)}")
    print(f"Validação: {len(X_validation)}")
    print(f"Teste: {len(X_test)}")

    # Baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_metrics = evaluate_model(
        "Baseline",
        baseline,
        X_test,
        y_test,
    )

    # Pipeline de Machine Learning
    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTreinando Regressão Logística...")

    model.fit(X_train, y_train)

    # Escolha do threshold usando SOMENTE validação
    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    threshold_results = []

    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:

        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        threshold_results.append({
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        })

    print("\n--- Análise de thresholds (validação) ---")

    for result in threshold_results:
        print(
            f"Threshold: {result['threshold']:.2f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f} | "
            f"F1: {result['f1_score']:.4f}"
        )

    # Escolhe o threshold com maior F1 na validação
    best_result = max(
        threshold_results,
        key=lambda result: result["f1_score"],
    )

    best_threshold = best_result["threshold"]

    print(f"\nThreshold escolhido: {best_threshold:.2f}")

    # Avaliação final no conjunto de teste
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\n--- Matriz de confusão (teste) ---")

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    print(matrix)

    final_metrics = evaluate_model(
        "Regressão Logística ajustada",
        model,
        X_test,
        y_test,
        threshold=best_threshold,
    )

    final_metrics["threshold"] = best_threshold

    # Salvar artefatos
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "logistic_regression.joblib"
    results_path = RESULTS_DIR / "model_comparison.json"

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
        },
        model_path,
    )

    results = {
        "baseline": baseline_metrics,
        "logistic_regression": final_metrics,
        "threshold_analysis": threshold_results,
        "dataset": {
            "total_records": len(X),
            "training_records": len(X_train),
            "validation_records": len(X_validation),
            "test_records": len(X_test),
            "features": X.shape[1],
            "random_state": RANDOM_STATE,
        },
    }

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n--- Arquivos gerados ---")
    print(f"Modelo: {model_path}")
    print(f"Resultados: {results_path}")

    return results


if __name__ == "__main__":
    train()
