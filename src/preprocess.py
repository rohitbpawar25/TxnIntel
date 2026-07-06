import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicate transactions and sorts chronologically by Time."""
    df_clean = df.drop_duplicates().reset_index(drop=True)
    df_clean = df_clean.sort_values("Time").reset_index(drop=True)
    return df_clean

def split_chronologically(df: pd.DataFrame, train_ratio: float = 0.8):
    """Splits data chronologically based on Time index."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def build_pipeline() -> Pipeline:
    """Returns the preprocessing pipeline mirroring Part A/B standard configuration."""
    pca_cols = [f"V{i}" for i in range(1, 29)]
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', RobustScaler(), ['Time', 'Amount']),
            ('imputer', SimpleImputer(strategy='median'), pca_cols)
        ]
    )
    return Pipeline([
        ('preprocessor', preprocessor)
    ])

