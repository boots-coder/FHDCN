"""
Utility functions for feature partitioning and recall.
"""

import numpy as np
from typing import Dict


def split_features_by_paper(
        X: np.ndarray,
        ranked_order: np.ndarray,
        ratio1: float = 0.05,
        ratio2: float = 0.10,
        ratio3: float = 0.10
) -> Dict[int, np.ndarray]:
    """
    Partition features into 4 levels based on ranking (Eq.3-6).

        f^1: top p1 features
        f^2: next p2 features
        f^3: next p3 features
        f^4: remaining features (recall pool)

    Args:
        X: data matrix (n_samples, n_features)
        ranked_order: feature indices sorted by importance
        ratio1, ratio2, ratio3: proportion for f^1, f^2, f^3

    Returns:
        Dict mapping level (1-4) to feature submatrix
    """
    n_features = X.shape[1]

    K1 = max(1, int(ratio1 * n_features))
    K2 = max(1, int(ratio2 * n_features))
    K3 = max(1, int(ratio3 * n_features))

    if K1 + K2 + K3 > n_features:
        K3 = max(1, n_features - K1 - K2)

    indices_f1 = ranked_order[:K1]
    indices_f2 = ranked_order[K1:K1+K2]
    indices_f3 = ranked_order[K1+K2:K1+K2+K3]
    indices_f4 = ranked_order[K1+K2+K3:]

    feature_levels = {
        1: X[:, indices_f1],
        2: X[:, indices_f2],
        3: X[:, indices_f3],
        4: X[:, indices_f4]
    }

    print(f"\n[Feature Partitioning]")
    print(f"  Total features: {n_features}")
    print(f"  Ratios: p1={ratio1:.0%}, p2={ratio2:.0%}, p3={ratio3:.0%}")
    print(f"  Counts: K1={K1}, K2={K2}, K3={K3}")
    print(f"  f^1: {len(indices_f1)} features (rank 1-{K1}, {100*len(indices_f1)/n_features:.1f}%)")
    print(f"  f^2: {len(indices_f2)} features (rank {K1+1}-{K1+K2}, {100*len(indices_f2)/n_features:.1f}%)")
    print(f"  f^3: {len(indices_f3)} features (rank {K1+K2+1}-{K1+K2+K3}, {100*len(indices_f3)/n_features:.1f}%)")
    print(f"  f^4: {len(indices_f4)} features (rank {K1+K2+K3+1}-{n_features}, {100*len(indices_f4)/n_features:.1f}%) [recall pool]")

    return feature_levels


def recall_f4_features(
        f4_features: np.ndarray,
        recall_ratio: float,
        random_state: int = None,
        data_type: str = None
) -> np.ndarray:
    """
    Randomly recall k features from f^4 (Section 3.2).
    k = floor(r * |f^4|)

    Args:
        f4_features: feature matrix of f^4 (n_samples, n_f4)
        recall_ratio: recall ratio r in (0, 1)
        random_state: random seed
        data_type: label for logging

    Returns:
        Recalled feature matrix (n_samples, k)
    """
    n_samples, n_f4 = f4_features.shape

    k = max(1, min(int(recall_ratio * n_f4), n_f4))

    if random_state is not None:
        np.random.seed(random_state)

    recalled_indices = np.random.choice(n_f4, size=k, replace=False)
    recalled_features = f4_features[:, recalled_indices]

    if data_type:
        print(f"  [Recall-{data_type}] Sampled {k} features from f^4 ({n_f4} total, r={recall_ratio:.2f})")
    else:
        print(f"  [Recall] Sampled {k} features from f^4 ({n_f4} total, r={recall_ratio:.2f})")

    return recalled_features
