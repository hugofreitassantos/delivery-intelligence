
import pandas as pd

from src.delivery_intelligence.data import load_data


TARGET = "is_late"

NUMERICAL_FEATURES = [
    "n_items",
    "n_distinct_products",
    "n_sellers",
    "total_price",
    "total_freight",
    "max_item_price",
    "total_weight_g",
    "max_volume_cm3",
    "avg_photos_qty",
    "avg_description_len",
    "distance_km",
    "purchase_year",
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "purchase_is_weekend",
    "purchase_week_of_year",
    "estimated_delivery_days",
    "freight_ratio",
    "price_per_item",
    "freight_per_kg",
    "same_state",
    "is_sao_paulo_seller",
    "multi_seller_order",
    "weight_per_item_g",
]

CATEGORICAL_FEATURES = [
    "primary_category",
    "customer_state",
    "seller_state",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Variáveis que não devem participar da previsão
EXCLUDED_FEATURES = [
    "actual_delivery_days",
    "days_late",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_status",
    "n_payment_methods",
    "total_payment_value",
    "max_installments",
    "payment_type",
]


def prepare_features():
    """Seleciona os atributos e a variável-alvo."""

    df = load_data()

    # Verifica se todas as colunas necessárias existem
    required_columns = FEATURE_COLUMNS + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colunas não encontradas: {missing_columns}"
        )

    # Remove registros sem variável-alvo
    df = df.dropna(subset=[TARGET]).copy()

    # Valida a variável-alvo
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("A variável-alvo deve conter 0 e 1.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int).copy()

    return X, y


if __name__ == "__main__":
    X, y = prepare_features()

    print("\n--- Preparação dos dados ---")
    print(f"Registros: {len(X)}")
    print(f"Quantidade de atributos: {X.shape[1]}")

    print("\n--- Variáveis numéricas ---")
    print(len(NUMERICAL_FEATURES))

    print("\n--- Variáveis categóricas ---")
    print(len(CATEGORICAL_FEATURES))

    print("\n--- Valores ausentes nos atributos ---")
    print(X.isna().sum()[X.isna().sum() > 0])

    print("\n--- Distribuição do target ---")
    print(y.value_counts(normalize=True))
