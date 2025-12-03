#!/usr/bin/env python
"""
Simple baseline benchmarking on concatenated npy features.

Expected files in --data_root:
    cog.npy   : (N, d_cog)
    morph.npy : (N, d_morph)
    sc.npy    : (N, d_sc)
    fc.npy    : (N, d_fc)
    labels.npy: (N,) or (N, 1)  -- continuous target

Models:
    - Lasso regression
    - Support Vector Regression (SVR)
    - K Nearest Neighbors regression (KNN)
    - KMeans clustering (cluster -> mean-label regressor)

Usage:
    python ml_baselines.py --data_root ./npy_data --out_dir ./results_baselines
"""

import os
import json
import argparse

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.multioutput import MultiOutputRegressor


# ---------------------- data utilities ---------------------- #

def load_concatenated_features(data_root: str):
    """
    Load cog, morph, sc, fc features and labels, and concatenate along feature axis.

    Any feature with ndim > 2 is flattened over all axes after the first:
        (N, d1, d2, ...) -> (N, d1 * d2 * ...)

    Returns
    -------
    X : np.ndarray, shape (N, D_total)
    y : np.ndarray, shape (N,)
    """
    feat_names = ["cog", "morph", "sc", "fc"]
    mats = []

    for name in feat_names:
        path = os.path.join(data_root, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")

        arr = np.load(path)
        raw_shape = arr.shape

        # Ensure 2D (N, D_flat)
        if arr.ndim == 1:
            # (N,) -> (N, 1)
            arr = arr[:, None]
        elif arr.ndim > 2:
            # (N, d1, d2, ...) -> (N, d1*d2*...)
            n = arr.shape[0]
            arr = arr.reshape(n, -1)

        print(f"[debug] {name}.npy raw shape {raw_shape} -> flat shape {arr.shape}")
        mats.append(arr)

    # Now all mats[i] are 2D: (N, D_i)
    X = np.concatenate(mats, axis=1)

    labels_path = os.path.join(data_root, "labels.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    y = np.load(labels_path).astype(np.float64)

    # Handle labels shape:
    # - If already 2D and first dim matches X, use as-is.
    # - If 1D and length is a multiple of N, reshape to (N, n_targets).
    N = X.shape[0]

    if y.ndim == 1:
        if y.shape[0] == N:
            y = y.reshape(N, 1)
        elif y.shape[0] % N == 0:
            n_targets = y.shape[0] // N
            print(f"[debug] inferring multi-output labels: reshaping from {y.shape} -> ({N}, {n_targets})")
            y = y.reshape(N, n_targets)
        else:
            raise ValueError(
                f"Cannot reshape labels of shape {y.shape} to match N={N}. "
                f"Expected length to be N * n_targets."
            )
    elif y.ndim == 2:
        if y.shape[0] != N:
            raise ValueError(f"Feature/label size mismatch: X has {N} rows, y has {y.shape[0]}")
    else:
        raise ValueError(f"Unsupported labels ndim={y.ndim}, shape={y.shape}")

    print(f"[debug] final X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def make_splits(X, y, val_size=0.15, test_size=0.15, seed=0):
    """
    Create train/val/test splits.

    val_size and test_size are fractions of the total dataset.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Re-normalize val fraction relative to remaining data
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=rel_val, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Standard-scale features using train statistics only.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ---------------------- metrics ---------------------- #

def compute_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}


# ---------------------- model runners ---------------------- #

def run_lasso(X_train, y_train, X_val, y_val, X_test, y_test, alpha=1.0):
    base = Lasso(alpha=alpha, max_iter=10000)
    model = MultiOutputRegressor(base)

    model.fit(X_train, y_train)

    y_hat_tr = model.predict(X_train)
    y_hat_va = model.predict(X_val)
    y_hat_te = model.predict(X_test)

    return {
        "train": compute_regression_metrics(y_train, y_hat_tr),
        "val": compute_regression_metrics(y_val, y_hat_va),
        "test": compute_regression_metrics(y_test, y_hat_te),
        "model_params": {
            "alpha": alpha,
        }
    }


def run_svr(X_train, y_train, X_val, y_val, X_test, y_test,
            C=10.0, epsilon=0.1, kernel="rbf"):

    base = SVR(C=C, epsilon=epsilon, kernel=kernel)
    model = MultiOutputRegressor(base)

    model.fit(X_train, y_train)

    y_hat_tr = model.predict(X_train)
    y_hat_va = model.predict(X_val)
    y_hat_te = model.predict(X_test)

    return {
        "train": compute_regression_metrics(y_train, y_hat_tr),
        "val": compute_regression_metrics(y_val, y_hat_va),
        "test": compute_regression_metrics(y_test, y_hat_te),
        "model_params": {
            "C": C,
            "epsilon": epsilon,
            "kernel": kernel,
        }
    }


def run_knn(X_train, y_train, X_val, y_val, X_test, y_test,
            n_neighbors=10, weights="distance"):
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)

    y_hat_tr = model.predict(X_train)
    y_hat_va = model.predict(X_val)
    y_hat_te = model.predict(X_test)

    return {
        "train": compute_regression_metrics(y_train, y_hat_tr),
        "val": compute_regression_metrics(y_val, y_hat_va),
        "test": compute_regression_metrics(y_test, y_hat_te),
        "model_params": {
            "n_neighbors": n_neighbors,
            "weights": weights,
        }
    }


def run_kmeans_regression(X_train, y_train, X_val, y_val, X_test, y_test,
                          n_clusters=3, random_state=0):
    """
    Unsupervised KMeans on X, then a simple regressor:
    each cluster predicts the mean label(s) of its training members.

    Works for single- or multi-output y.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X_train)

    train_clusters = km.labels_

    # y can be (N,) or (N, n_targets)
    if y_train.ndim == 1:
        cluster_means = np.zeros(n_clusters, dtype=np.float64)
    else:
        cluster_means = np.zeros((n_clusters, y_train.shape[1]), dtype=np.float64)

    for k in range(n_clusters):
        mask = (train_clusters == k)
        if mask.sum() == 0:
            # fallback: global mean
            cluster_means[k] = y_train.mean(axis=0) if y_train.ndim > 1 else y_train.mean()
        else:
            if y_train.ndim == 1:
                cluster_means[k] = y_train[mask].mean()
            else:
                cluster_means[k] = y_train[mask].mean(axis=0)

    def predict_with_cluster_means(X):
        cluster_idx = km.predict(X)
        return cluster_means[cluster_idx]

    y_hat_tr = predict_with_cluster_means(X_train)
    y_hat_va = predict_with_cluster_means(X_val)
    y_hat_te = predict_with_cluster_means(X_test)

    inertia = float(km.inertia_)
    try:
        sil = float(silhouette_score(X_train, train_clusters))
    except Exception:
        sil = None

    out = {
        "train": compute_regression_metrics(y_train, y_hat_tr),
        "val": compute_regression_metrics(y_val, y_hat_va),
        "test": compute_regression_metrics(y_test, y_hat_te),
        "model_params": {
            "n_clusters": n_clusters,
            "random_state": random_state,
        },
        "unsupervised_metrics": {
            "inertia": inertia,
            "silhouette_train": sil,
        }
    }
    return out

#/Users/ethanmeidinger/GNNproject/GNNproject/data/av/common

# ---------------------- main ---------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Simple ML baselines on concatenated npy features.")
    p.add_argument("--data_root", type=str, default="/Users/ethanmeidinger/GNNproject/GNNproject/data/av/common",
                   help="Folder containing cog.npy, morph.npy, sc.npy, fc.npy, labels.npy")
    p.add_argument("--out_dir", type=str, default="results_baselines",
                   help="Where to save JSON metrics.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)

    # Lasso
    p.add_argument("--lasso_alpha", type=float, default=0.1)

    # SVR
    p.add_argument("--svr_C", type=float, default=10.0)
    p.add_argument("--svr_epsilon", type=float, default=0.1)

    # KNN
    p.add_argument("--knn_k", type=int, default=10)

    # KMeans
    p.add_argument("--kmeans_k", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] Loading features from: {args.data_root}")
    X, y = load_concatenated_features(args.data_root)
    print(f"[info] Loaded X shape: {X.shape}, y shape: {y.shape}")

    #features have already been normalized across age and gender, meaning we should not scale again
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        X, y, val_size=args.val_frac, test_size=args.test_frac, seed=args.seed
    )

    print(f"[info] Train/Val/Test sizes: {X_train.shape[0]} / {X_val.shape[0]} / {X_test.shape[0]}")

    results = {}

    # --------- Lasso --------- #
    print("\n========== LASSO ==========")
    res_lasso = run_lasso(
        X_train, y_train, X_val, y_val, X_test, y_test,
        alpha=args.lasso_alpha
    )
    results["lasso"] = res_lasso
    for split in ["train", "val", "test"]:
        m = res_lasso[split]
        print(f"[lasso] {split}: MSE={m['mse']:.4f} | MAE={m['mae']:.4f} | R2={m['r2']:.4f}")

    # --------- SVR --------- #
    print("\n========== SVR ==========")
    res_svr = run_svr(
        X_train, y_train, X_val, y_val, X_test, y_test,
        C=args.svr_C, epsilon=args.svr_epsilon
    )
    results["svr"] = res_svr
    for split in ["train", "val", "test"]:
        m = res_svr[split]
        print(f"[svr] {split}: MSE={m['mse']:.4f} | MAE={m['mae']:.4f} | R2={m['r2']:.4f}")

    # --------- KNN --------- #
    print("\n========== KNN ==========")
    res_knn = run_knn(
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_neighbors=args.knn_k
    )
    results["knn"] = res_knn
    for split in ["train", "val", "test"]:
        m = res_knn[split]
        print(f"[knn] {split}: MSE={m['mse']:.4f} | MAE={m['mae']:.4f} | R2={m['r2']:.4f}")

    # --------- KMeans (cluster -> mean-label regressor) --------- #
    print("\n========== KMEANS (cluster-mean regressor) ==========")
    res_km = run_kmeans_regression(
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_clusters=args.kmeans_k, random_state=args.seed
    )
    results["kmeans"] = res_km
    for split in ["train", "val", "test"]:
        m = res_km[split]
        print(f"[kmeans] {split}: MSE={m['mse']:.4f} | MAE={m['mae']:.4f} | R2={m['r2']:.4f}")
    unsup = res_km["unsupervised_metrics"]
    print(f"[kmeans] inertia={unsup['inertia']:.4f}, "
          f"silhouette_train={unsup['silhouette_train'] if unsup['silhouette_train'] is not None else 'nan'}")

    # --------- Save JSON --------- #
    out_path = os.path.join(args.out_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] Saved results to {out_path}")


if __name__ == "__main__":
    main()
