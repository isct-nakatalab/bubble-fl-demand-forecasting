import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def compute_feature_importance(features, labels, test_size=0.2, random_state=42):
    """Compute XGBoost feature importance for a single client's data.

    Args:
        features: np.ndarray of shape (n_samples, n_features)
        labels: np.ndarray of shape (n_samples,)

    Returns:
        importance_scores: np.ndarray of shape (n_features,)
    """
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    labels = labels.reshape(-1)

    X_train, _, y_train, _ = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    importance_scores = model.feature_importances_
    return importance_scores


def compute_all_feature_importances(datasets, client_ids):
    """Compute feature importance for all active clients.

    Args:
        datasets: dict of client_id -> {train_features, train_labels, ...}
        client_ids: list of client IDs

    Returns:
        importance_matrix: np.ndarray of shape (n_clients, n_features)
        active_client_ids: list of client IDs
    """
    importance_matrix = []
    active_client_ids = []

    for client_id in client_ids:
        if client_id not in datasets:
            continue
        data = datasets[client_id]
        features = data["train_features"].numpy()
        labels = data["train_labels"].numpy()

        scores = compute_feature_importance(features, labels)
        importance_matrix.append(scores)
        active_client_ids.append(client_id)

    return np.array(importance_matrix), active_client_ids
