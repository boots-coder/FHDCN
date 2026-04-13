"""
Feature Hierarchical Module (FHM) - Paper Section 3.1

Hybrid feature ranking strategy:
1. F-test (ANOVA) for statistical significance -> R_stat (Eq.1)
2. Lasso regression for model contribution -> R_lasso (Eq.2)
3. Borda count fusion: S_i = R_stat^(i) + R_lasso^(i)

Features are sorted by S_i in ascending order to obtain L_All,
then partitioned into f^1, f^2, f^3, f^4 (Eq.3-6).
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from typing import Dict


class PaperFeatureRanker:
    """
    Feature ranking via F-test + Lasso + Borda count.

    Usage:
        ranker = PaperFeatureRanker()
        ranker.fit(X_train, y_train)
        ranked_order = ranker.get_ranked_order()
        importance_scores = ranker.get_importance_scores()
    """

    def __init__(self, lasso_alpha=None, max_iter=2000, random_state=43, verbose=True):
        self.lasso_alpha = lasso_alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.ranked_order_ = None
        self.borda_scores_ = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        if self.verbose:
            print(f"\n[PaperFeatureRanker] Starting feature ranking...")
            print(f"  Samples: {n_samples}, Features: {n_features}, Classes: {len(np.unique(y))}")

        # Step 1: F-test (Eq.1)
        if self.verbose:
            print(f"\n  Step 1: Computing F-test statistics...")

        f_scores, _ = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

        R_stat = np.empty(n_features, dtype=int)
        for rank, feat_idx in enumerate(np.argsort(-f_scores), start=1):
            R_stat[feat_idx] = rank

        if self.verbose:
            print(f"    F-score range: [{f_scores.min():.4f}, {f_scores.max():.4f}]")
            print(f"    Top-5 features (F-test): {np.argsort(-f_scores)[:5]}")

        # Step 2: Lasso regression (Eq.2)
        if self.verbose:
            print(f"\n  Step 2: Training Lasso regression...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_classes = len(np.unique(y))
        if n_classes == 2:
            if self.lasso_alpha is not None:
                from sklearn.linear_model import LogisticRegression
                lasso = LogisticRegression(
                    penalty='l1', C=1.0 / self.lasso_alpha, solver='liblinear',
                    max_iter=self.max_iter, random_state=self.random_state
                )
            else:
                from sklearn.linear_model import LogisticRegressionCV
                lasso = LogisticRegressionCV(
                    penalty='l1', cv=min(5, n_samples // 10) if n_samples >= 20 else 3,
                    solver='liblinear', max_iter=self.max_iter,
                    random_state=self.random_state, n_jobs=1
                )
            lasso.fit(X_scaled, y)
            lasso_coefs = np.abs(lasso.coef_).flatten()
        else:
            if self.lasso_alpha is not None:
                from sklearn.linear_model import LogisticRegression
                lasso = LogisticRegression(
                    penalty='l1', C=1.0 / self.lasso_alpha, solver='liblinear',
                    multi_class='ovr', max_iter=self.max_iter,
                    random_state=self.random_state
                )
            else:
                from sklearn.linear_model import LogisticRegressionCV
                lasso = LogisticRegressionCV(
                    penalty='l1', cv=min(5, n_samples // 10) if n_samples >= 20 else 3,
                    solver='liblinear', multi_class='ovr', max_iter=self.max_iter,
                    random_state=self.random_state, n_jobs=1
                )
            lasso.fit(X_scaled, y)
            lasso_coefs = np.abs(lasso.coef_).max(axis=0)

        R_lasso = np.empty(n_features, dtype=int)
        for rank, feat_idx in enumerate(np.argsort(-lasso_coefs), start=1):
            R_lasso[feat_idx] = rank

        if self.verbose:
            n_selected = np.sum(lasso_coefs > 1e-6)
            print(f"    Lasso selected features: {n_selected}/{n_features}")
            print(f"    Coefficient range: [{lasso_coefs.min():.6f}, {lasso_coefs.max():.6f}]")
            print(f"    Top-5 features (Lasso): {np.argsort(-lasso_coefs)[:5]}")

        # Step 3: Borda count fusion
        if self.verbose:
            print(f"\n  Step 3: Borda count fusion...")

        borda_scores = R_stat + R_lasso
        ranked_order = np.argsort(borda_scores)

        if self.verbose:
            print(f"    Borda score range: [{borda_scores.min()}, {borda_scores.max()}]")
            print(f"    Top-10 features (final): {ranked_order[:10]}")

            print(f"\n  Top-10 feature details:")
            print(f"    {'Rank':<6} {'FeatureID':<12} {'F-Rank':<10} {'Lasso-Rank':<12} {'Borda-Score':<12}")
            for i, feat_idx in enumerate(ranked_order[:10], start=1):
                print(f"    {i:<6} {feat_idx:<12} {R_stat[feat_idx]:<10} "
                      f"{R_lasso[feat_idx]:<12} {borda_scores[feat_idx]:<12}")

        self.borda_scores_ = borda_scores
        self.ranked_order_ = ranked_order
        self._fitted = True

        if self.verbose:
            print(f"\n[PaperFeatureRanker] Feature ranking complete!")

        return self

    def get_ranked_order(self) -> np.ndarray:
        """Return feature indices sorted by importance (most to least)."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self.ranked_order_

    def get_importance_scores(self) -> np.ndarray:
        """Return Borda scores (lower = more important)."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self.borda_scores_
