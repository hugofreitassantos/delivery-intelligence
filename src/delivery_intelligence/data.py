
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "olist_analytical.parquet"


def load_data() -> pd.DataFrame:
    """Carrega os dados históricos de entregas."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    if df.empty:
        raise ValueError("O dataset está vazio.")

    return df


if __name__ == "__main__":
    df = load_data()

    print("\n--- Dimensões ---")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    print("\n--- Primeiras linhas ---")
    print(df.head())

    print("\n--- Tipos de dados ---")
    print(df.dtypes)

    print("\n--- Valores ausentes ---")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    print("\n--- Distribuição da variável-alvo ---")
    print(df["is_late"].value_counts(dropna=False))
    print(df["is_late"].value_counts(normalize=True, dropna=False))
