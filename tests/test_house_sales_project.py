import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from house_sales_project import (  # noqa: E402
    clean_house_sales,
    make_model_pipeline,
    prepare_model_features,
    summarize_prices_by_bedrooms,
)


def sample_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "house_id": [1, 2, 3, 4],
            "city": ["Silvertown", None, "Riverford", "Poppleton"],
            "sale_price": [100000, 150000, 200000, 250000],
            "sale_date": ["2023-02-01", None, "2023-03-01", "2023-04-01"],
            "months_listed": [2.0, None, 4.0, 5.0],
            "bedrooms": [2, 3, None, 5],
            "house_type": ["Det.", "Semi", None, "Terr."],
            "area": ["100.0 sq.m.", "150.0 sq.m.", None, "250.0 sq.m."],
        }
    )


def test_cleaning_imputes_and_standardizes_values():
    cleaned = clean_house_sales(sample_sales())
    assert cleaned.shape == (4, 8)
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.loc[1, "city"] == "Unknown"
    assert cleaned.loc[0, "house_type"] == "Detached"
    assert cleaned.loc[1, "house_type"] == "Semi-detached"
    assert cleaned.loc[3, "house_type"] == "Terraced"
    assert str(cleaned.loc[1, "sale_date"].date()) == "2023-01-01"


def test_bedroom_summary_has_expected_columns_and_values():
    cleaned = clean_house_sales(sample_sales())
    summary = summarize_prices_by_bedrooms(cleaned)
    assert list(summary.columns) == ["bedrooms", "avg_price", "var_price"]
    assert summary["avg_price"].notna().all()


def test_model_pipeline_handles_missing_categorical_values():
    train = pd.DataFrame(
        {
            "city": pd.Series(["A", None, "B", "A"], dtype="string"),
            "area": [80.0, 100.0, np.nan, 140.0],
        }
    )
    valid = pd.DataFrame(
        {
            "city": pd.Series(["C", None], dtype="string"),
            "area": [90.0, np.nan],
        }
    )
    X_train, _ = prepare_model_features(train)
    X_valid, _ = prepare_model_features(valid)
    X_valid = X_valid.reindex(columns=X_train.columns)
    for estimator in (LinearRegression(), RandomForestRegressor(n_estimators=10, random_state=42)):
        pipeline = make_model_pipeline(X_train, estimator)
        pipeline.fit(X_train, [100.0, 120.0, 150.0, 180.0])
        predictions = pipeline.predict(X_valid)
        assert len(predictions) == 2
        assert np.isfinite(predictions).all()
