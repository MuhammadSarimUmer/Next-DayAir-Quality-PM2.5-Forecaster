"""
Task 8: Preprocessing pipeline with scaler + encoder in a ColumnTransformer.

This module builds sklearn transformers that:
- Scale numeric features (StandardScaler)
- Encode categorical features (OneHotEncoder for city)

The preprocessor is fitted on training data only, then applied to val/test.
Numeric features use median imputation (CAMS/weather gaps at series boundaries).
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor(numeric_cols, categorical_cols=['city']):
    """
    Build preprocessing ColumnTransformer.
    
    Parameters
    ----------
    numeric_cols : list of str
        Column names of numeric features to scale.
    categorical_cols : list of str, default=['city']
        Column names of categorical features to encode.
    
    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor. Fit this on training data, then transform val/test.
    
    Examples
    --------
    >>> from src.features import FEATURE_COLS
    >>> numeric_cols = [c for c in FEATURE_COLS if c != 'city']
    >>> preprocessor = build_preprocessor(numeric_cols)
    >>> preprocessor.fit(X_train)
    >>> X_train_transformed = preprocessor.transform(X_train)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def build_pipeline(model, numeric_cols, categorical_cols=['city']):
    """
    Convenience function: preprocessor + model in one Pipeline.
    
    This creates a full sklearn Pipeline that:
    1. Preprocesses features (scale + encode)
    2. Fits the model
    
    Parameters
    ----------
    model : sklearn estimator
        Any sklearn-compatible model (e.g., Ridge(), RandomForestRegressor()).
    numeric_cols : list of str
        Column names of numeric features to scale.
    categorical_cols : list of str, default=['city']
        Column names of categorical features to encode.
    
    Returns
    -------
    Pipeline
        Unfitted pipeline. Call .fit(X, y) on training data.
    
    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from src.features import FEATURE_COLS
    >>> numeric_cols = [c for c in FEATURE_COLS if c != 'city']
    >>> pipeline = build_pipeline(Ridge(alpha=1.0), numeric_cols)
    >>> pipeline.fit(X_train, y_train)
    >>> y_pred = pipeline.predict(X_val)
    """
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor(numeric_cols, categorical_cols)),
        ('model', model)
    ])
    
    return pipeline


if __name__ == '__main__':
    """
    Test the preprocessor with dummy data.
    Run: python -m src.pipeline
    """
    import pandas as pd
    import numpy as np
    from src.features import FEATURE_COLS
    
    print("=" * 60)
    print("Task 8: Testing Preprocessing Pipeline")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    
    dummy_data = pd.DataFrame({
        'pm25_now': np.random.uniform(10, 200, n_samples),
        'pm25_lag_1h': np.random.uniform(10, 200, n_samples),
        'temperature_2m': np.random.uniform(20, 40, n_samples),
        'relative_humidity_2m': np.random.uniform(30, 90, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'city': np.random.choice(['karachi', 'lahore'], n_samples)
    })
    
    # Separate numeric and categorical
    numeric_cols = [c for c in dummy_data.columns if c != 'city']
    categorical_cols = ['city']
    
    print(f"\nInput shape: {dummy_data.shape}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Build and fit preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    transformed = preprocessor.fit_transform(dummy_data)
    
    print(f"\nTransformed shape: {transformed.shape}")
    print(f"Expected features: {len(numeric_cols)} numeric + 1 encoded city = {len(numeric_cols) + 1}")
    
    # Check scaling worked (mean ≈ 0, std ≈ 1 for numeric features)
    print(f"\nFirst numeric feature after scaling:")
    print(f"  Mean: {transformed[:, 0].mean():.4f} (should be ≈ 0)")
    print(f"  Std:  {transformed[:, 0].std():.4f} (should be ≈ 1)")
    
    # Check encoding worked (last column should be 0 or 1)
    print(f"\nCity encoding (last column):")
    print(f"  Unique values: {np.unique(transformed[:, -1])}")
    print(f"  (0 = karachi, 1 = lahore)")
    
    print("\n" + "=" * 60)
    print("Preprocessor test complete")
    print("=" * 60)
