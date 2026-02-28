"""
Training script — loads datasets, trains XGBoost classifier, saves model.
Run this first thing Saturday morning once private dataset scope is confirmed.

Usage:
    python classifier/train.py --data data/public --out classifier/model.pkl
"""
import argparse
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib


def load_dataset(data_dir: str):
    """
    TODO: Load features + labels from data_dir.
    Expected format: numpy .npz files with keys 'X' (features) and 'y' (string labels).
    Adjust once private dataset format is known.
    """
    raise NotImplementedError(f"Load dataset from {data_dir}")


def train(data_dir: str, output_path: str):
    X, y = load_dataset(data_dir)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )

    scores = cross_val_score(model, X, y_enc, cv=5, scoring="roc_auc_ovr")
    print(f"CV AUC (OVR): {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X, y_enc)
    joblib.dump({"model": model, "label_encoder": le}, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public")
    parser.add_argument("--out", default="classifier/model.pkl")
    args = parser.parse_args()
    train(args.data, args.out)
