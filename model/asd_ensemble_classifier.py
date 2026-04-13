"""
ASD Adaptive Ensemble Classification Module (AECM) - Paper Section 3.4

Core scoring function (Eq.8):
    Score(C) = w1 * A(C) + w2 * S(C) + w3 * D(C)

Where:
    A(C): mean cross-validation accuracy (Eq.9)
    S(C): stability score = 1 - sigma_A(C) / max_{c' in P} sigma_A(c') (Eq.10)
    D(C): diversity score = exp(-lambda * N_cat(C)) (Eq.12)

Weighted voting prediction (Eq.13-15):
    omega_i = Score(C_i) / sum_j Score(C_j)
    P(y=c|x) = sum_i omega_i * P_i(y=c|x)
"""

import numpy as np
from copy import deepcopy
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class Candidate:
    """Wrapper for a candidate base classifier."""

    def __init__(self, name, model, acc=None, std=None):
        self.name = name
        self.model = model
        self.category = self._infer_category(model)
        self.acc = acc        # A(C): mean CV accuracy
        self.std = std        # sigma_A(C): std of CV accuracy
        self.n_cat = 0        # N_cat(C): count of same-category models in selected set
        self.stability = None  # S(C): computed after evaluating all candidates

    def _infer_category(self, model):
        """Infer algorithm family from model type."""
        model_type = type(model).__name__
        if 'SVC' in model_type or 'SVM' in model_type:
            return 'SVM'
        elif 'RandomForest' in model_type:
            return 'RF'
        elif 'ExtraTrees' in model_type:
            return 'ExtraTrees'
        elif 'GradientBoosting' in model_type:
            return 'GBM'
        elif 'KNeighbors' in model_type:
            return 'KNN'
        elif 'GaussianNB' in model_type or 'BernoulliNB' in model_type:
            return 'NB'
        elif 'LogisticRegression' in model_type:
            return 'LR'
        else:
            return 'Other'

    def calculate_score(self, w1, w2, w3, lambda_param):
        """
        ASD score (Eq.8):
        Score(C) = w1 * A(C) + w2 * S(C) + w3 * D(C)
        """
        diversity = np.exp(-lambda_param * self.n_cat)  # D(C), Eq.12
        return w1 * self.acc + w2 * self.stability + w3 * diversity


def build_model_pool():
    """
    Build heterogeneous candidate classifier pool (Section 4.2.2).

    12 classifiers: SVM (Linear, RBF), KNN (K=3,5,7),
    GaussianNB, BernoulliNB, RF, ExtraTrees, GBM, LR (L1, L2).
    """
    models = [
        # SVM family
        ('SVM_linear', SVC(kernel='linear', C=1.0, probability=True, random_state=42)),
        ('SVM_rbf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
        # KNN family (K=3, 5, 7)
        ('KNN_k3', KNeighborsClassifier(n_neighbors=3)),
        ('KNN_k5', KNeighborsClassifier(n_neighbors=5)),
        ('KNN_k7', KNeighborsClassifier(n_neighbors=7)),
        # Naive Bayes family
        ('GaussianNB', GaussianNB()),
        ('BernoulliNB', BernoulliNB()),
        # Ensemble methods
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('ExtraTrees', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('GBM', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        # Logistic Regression (L1/L2)
        ('LR_L1', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)),
        ('LR_L2', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)),
    ]
    return models


class ASDWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    ASD weighted voting ensemble classifier.

    Uses Algorithm 1 (ASD Adaptive Screening) to select the optimal
    base classifier combination, then performs weighted voting prediction
    based on normalized ASD scores (Eq.13-15).
    """

    def __init__(self, n_select=5, w1=2.0, w2=0.1, w3=0.5, lambda_param=1.5, cv=5):
        self.n_select = n_select
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.lambda_param = lambda_param
        self.cv = cv
        self.selected_models_ = None
        self.asd_scores_ = None
        self.weights_ = None
        self.classes_ = None
        self.scaler_ = None

    def fit(self, X, y, model_pool=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        if model_pool is None:
            model_pool = build_model_pool()

        # Evaluate all candidates via k-fold CV
        candidates = []
        for name, model in model_pool:
            scores = cross_val_score(model, X_scaled, y, cv=self.cv, scoring='accuracy')
            candidate = Candidate(name=name, model=model, acc=scores.mean(), std=scores.std())
            candidates.append(candidate)

        # Compute stability S(C) = 1 - sigma / max_sigma (Eq.10)
        max_std = max(c.std for c in candidates)
        for c in candidates:
            c.stability = 1.0 - c.std / max_std if max_std > 0 else 1.0

        # ASD iterative selection (Algorithm 1)
        selected = self._asd_select(candidates)

        # Compute voting weights (Eq.13)
        self.selected_models_ = []
        self.asd_scores_ = []
        self.selected_candidates_ = []

        for candidate in selected:
            score = candidate.calculate_score(self.w1, self.w2, self.w3, self.lambda_param)
            self.asd_scores_.append(score)
            self.selected_candidates_.append(candidate)
            model_clone = deepcopy(candidate.model)
            model_clone.fit(X_scaled, y)
            self.selected_models_.append(model_clone)

        # omega_i = Score(C_i) / sum Score(C_j)
        self.weights_ = np.array(self.asd_scores_)
        self.weights_ = self.weights_ / self.weights_.sum()

        return self

    def _asd_select(self, candidates):
        """Algorithm 1: ASD Adaptive Screening."""
        remaining = deepcopy(candidates)
        selected = []

        for _ in range(min(self.n_select, len(candidates))):
            # Line 4-5: compute Score(C) and select argmax
            scores = [(c, c.calculate_score(self.w1, self.w2, self.w3, self.lambda_param))
                      for c in remaining]
            scores.sort(key=lambda x: x[1], reverse=True)

            best_candidate, _ = scores[0]
            selected.append(best_candidate)  # Line 6

            # Line 7-8: update N_cat and remove selected
            remaining = [c for c in remaining if c.name != best_candidate.name]
            for c in remaining:
                if c.category == best_candidate.category:
                    c.n_cat += 1

        return selected

    def predict_proba(self, X):
        """Weighted soft voting (Eq.14)."""
        X = np.asarray(X)
        X_scaled = self.scaler_.transform(X)

        weighted_proba = np.zeros((X.shape[0], len(self.classes_)))
        for model, weight in zip(self.selected_models_, self.weights_):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(X_scaled)
                if len(self.classes_) == 2:
                    proba_1 = 1 / (1 + np.exp(-decision))
                    proba = np.column_stack([1 - proba_1, proba_1])
                else:
                    proba = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)
            else:
                pred = model.predict(X_scaled)
                proba = np.zeros((X.shape[0], len(self.classes_)))
                proba[np.arange(X.shape[0]), pred.astype(int)] = 1
            weighted_proba += weight * proba

        return weighted_proba

    def predict(self, X):
        """Final classification (Eq.15): argmax P(y=c|x)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_info(self):
        """Return ensemble details for inspection."""
        info = {'n_models': len(self.selected_models_), 'models': []}
        if hasattr(self, 'selected_candidates_'):
            for i, (model, score, weight, candidate) in enumerate(zip(
                self.selected_models_, self.asd_scores_, self.weights_, self.selected_candidates_
            )):
                info['models'].append({
                    'index': i + 1,
                    'name': candidate.name,
                    'asd_score': score,
                    'weight': weight,
                    'cv_acc': candidate.acc,
                    'cv_std': candidate.std,
                    'n_cat': candidate.n_cat,
                    'category': candidate.category
                })
        return info
