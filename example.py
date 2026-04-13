#!/usr/bin/env python3
"""
FHDCN - Quick Start Example
Runs the full cascade pipeline on the ALLAML dataset.

Usage:
    python example.py
"""
from __future__ import annotations
import os, random, warnings
import numpy as np
import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from model.feature_ranker_paper import PaperFeatureRanker
from model.asd_ensemble_classifier import ASDWeightedEnsemble
from model.extractor_BNN import ProgressiveBNNExtractor
from utils import split_features_by_paper, recall_f4_features

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 43
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ALLAML.mat")

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def load_mat(path):
    """Load a .mat dataset and return X (n x p), y (n,) with labels in {0, 1}."""
    mat = sio.loadmat(path, squeeze_me=True)
    mat = {k: v for k, v in mat.items() if not k.startswith("__")}

    X_keys = ["X", "data", "fea", "features", "expression"]
    Y_keys = ["Y", "y", "label", "labels", "gnd", "target", "class"]

    Xk = next((k for k in X_keys if k in mat), None)
    Yk = next((k for k in Y_keys if k in mat), None)
    if Xk is None:
        Xk = next(k for k, v in mat.items() if isinstance(v, np.ndarray) and v.ndim == 2)
    if Yk is None:
        Yk = next(k for k, v in mat.items() if isinstance(v, np.ndarray) and v.ndim == 1)

    X = mat[Xk].astype(np.float32)
    y = mat[Yk].astype(int).ravel()
    uniq = np.unique(y)
    y_bin = np.zeros_like(y)
    y_bin[y == uniq[1]] = 1
    return X, y_bin


def main():
    # ---- Load & split ----
    X, y = load_mat(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"Dataset: ALLAML  |  train={len(y_tr)}, test={len(y_te)}, features={X_tr.shape[1]}")

    # ---- Step 1: Feature Hierarchical Module (FHM) ----
    ranker = PaperFeatureRanker(verbose=True, random_state=SEED)
    ranker.fit(X_tr, y_tr)
    ranked = ranker.get_ranked_order()

    ratio1, ratio2, ratio3 = 0.05, 0.10, 0.10
    lvls_tr = split_features_by_paper(X_tr, ranked, ratio1, ratio2, ratio3)
    lvls_te = split_features_by_paper(X_te, ranked, ratio1, ratio2, ratio3)

    # ---- Step 2: Dynamic Cascade ----
    prev_tr = prev_te = None
    best_acc, plateau = 0.0, 0
    max_layers, patience = 20, 4
    recall_ratio = 0.1

    for layer in range(1, max_layers + 1):
        lvl = ((layer - 1) % 3) + 1  # mod-3 rotation
        h_tr, h_te = lvls_tr[lvl], lvls_te[lvl]

        # Feature Recall and Fusion Module (FRFM)
        if layer == 1:
            U_tr, U_te = h_tr, h_te
        else:
            recalled_tr = recall_f4_features(lvls_tr[4], recall_ratio, random_state=SEED + layer)
            recalled_te = recall_f4_features(lvls_te[4], recall_ratio, random_state=SEED + layer)
            prev_tr_np = prev_tr.cpu().numpy() if torch.is_tensor(prev_tr) else prev_tr
            prev_te_np = prev_te.cpu().numpy() if torch.is_tensor(prev_te) else prev_te
            U_tr = np.concatenate([h_tr, prev_tr_np, recalled_tr], axis=1)
            U_te = np.concatenate([h_te, prev_te_np, recalled_te], axis=1)

        # Gated Progressive Representation Module (GPRM)
        U_tr_t = torch.tensor(U_tr, dtype=torch.float32, device=DEVICE)
        U_te_t = torch.tensor(U_te, dtype=torch.float32, device=DEVICE)
        ext = ProgressiveBNNExtractor(U_tr.shape[1], len(np.unique(y_tr)), dropout=0.2).to(DEVICE)
        F_tr = ext.fit_transform(
            U_tr_t, torch.tensor(y_tr, dtype=torch.long, device=DEVICE),
            epochs=40, batch_size=min(64, len(y_tr)), verbose=False
        ).cpu().numpy()
        F_te = ext.transform(U_te_t).cpu().numpy()
        del ext, U_tr_t, U_te_t
        torch.cuda.empty_cache()

        # ASD Adaptive Ensemble Classification Module (AECM)
        clf = ASDWeightedEnsemble(n_select=3, w1=2.0, w2=0.1, w3=0.1, lambda_param=1.0, cv=5)
        clf.fit(F_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(F_te))

        print(f"Layer {layer:2d} (f^{lvl}) | dim={U_tr.shape[1]:5d} -> {F_tr.shape[1]} | Acc={acc:.4f}")

        if acc > best_acc + 1e-4:
            best_acc, plateau = acc, 0
        else:
            plateau += 1
        if plateau >= patience:
            print(f"Early stop at layer {layer} (no improvement for {patience} layers)")
            break

        prev_tr = torch.tensor(F_tr, dtype=torch.float32, device=DEVICE)
        prev_te = torch.tensor(F_te, dtype=torch.float32, device=DEVICE)
        del clf
        torch.cuda.empty_cache()

    print(f"\nBest test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
