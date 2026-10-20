import numpy as np
import pandas as pd

from ml_pipeline import build_features, haversine_km, minmax, train_and_compare


def test_haversine_distance_is_zero_for_same_point_and_known_for_one_degree():
    same = haversine_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    one_degree = haversine_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))

    assert same[0] == 0
    assert np.isclose(one_degree[0], 111.195, atol=0.01)


def test_feature_builder_adds_expected_features_without_target():
    source = pd.DataFrame(
        {
            "amount": [25.0, 40.0],
            "trans_time": pd.to_datetime(["2020-01-01 23:00", "2020-01-02 10:00"]),
            "cust_lat": [0.0, 1.0],
            "cust_long": [0.0, 1.0],
            "merch_lat": [0.0, 2.0],
            "merch_long": [0.0, 2.0],
            "category": ["shopping_net", "grocery_pos"],
            "dob": pd.to_datetime(["1990-01-01", "1980-01-01"]),
            "is_fraud": [0, 1],
        }
    )

    features = build_features(source)

    assert {"amount", "log_amount", "hour", "is_night", "day_of_week", "distance_km", "age"}.issubset(features.columns)
    assert "cat_shopping_net" in features.columns
    assert "cat_grocery_pos" in features.columns
    assert "is_fraud" not in features.columns
    assert features["is_night"].tolist() == [1, 0]


def test_minmax_clips_to_unit_interval():
    values = minmax(np.array([-2.0, 0.0, 2.0, 4.0]), 0.0, 2.0)

    np.testing.assert_allclose(values, [0.0, 0.0, 1.0, 1.0])


def test_model_selection_uses_validation_pr_auc_and_reports_held_out_rows():
    rng = np.random.default_rng(42)
    labels = pd.Series(np.array([0] * 200 + [1] * 50))
    signal = labels.to_numpy() * 2 + rng.normal(0, 0.5, size=len(labels))
    features = pd.DataFrame(
        {
            "signal": signal,
            "noise_a": rng.normal(size=len(labels)),
            "noise_b": rng.normal(size=len(labels)),
        }
    )

    best_name, _, _, results, _, evaluation = train_and_compare(features, labels)

    assert results["validation_pr_auc"].idxmax() == best_name
    assert results["selected"].sum() == 1
    assert results.loc[best_name, "selected"]
    assert len(evaluation["y_test"]) == 50
    assert all(len(scores) == 50 for scores in evaluation["scores"].values())
    assert results[["precision", "recall", "f1", "roc_auc", "pr_auc"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()
