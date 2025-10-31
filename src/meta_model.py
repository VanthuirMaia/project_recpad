import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
import pandas as pd

RANDOM_STATE = 42


def generate_oof_predictions(X, y, base_models, cv_splits=10):
    """
    Gera predições 'out-of-fold' (OOF) para cada modelo base.
    Essas predições servirão como entrada para o meta-aprendiz.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(X), len(base_models)))  # cada coluna = modelo
    trained_models = {}

    for j, (name, model) in enumerate(base_models.items()):
        print(f"\nTreinando modelo base: {name}")
        fold_preds = np.zeros(len(X))

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
            m = Pipeline(model.steps)  # clone
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = m.predict_proba(X.iloc[te_idx])[:, 1]
            fold_preds[te_idx] = preds
            print(f"  Fold {fold}: OK")

        oof_preds[:, j] = fold_preds
        model.fit(X, y)  # refit completo
        trained_models[name] = model

    oof_df = pd.DataFrame(oof_preds, columns=list(base_models.keys()), index=X.index)
    return oof_df, trained_models


def train_meta_model(oof_df, y, cv_splits=10):
    """
    Treina e avalia o meta-aprendiz (nível-1) usando as predições OOF.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    meta_scores = {"acc": [], "auc": [], "logloss": [], "brier": []}

    for fold, (tr, te) in enumerate(cv.split(oof_df, y), 1):
        meta = LogisticRegression(max_iter=200, solver="lbfgs", random_state=RANDOM_STATE)
        meta.fit(oof_df.iloc[tr], y.iloc[tr])
        p = meta.predict_proba(oof_df.iloc[te])[:, 1]
        yt = y.iloc[te]

        meta_scores["acc"].append(accuracy_score(yt, (p >= 0.5).astype(int)))
        meta_scores["auc"].append(roc_auc_score(yt, p))
        meta_scores["logloss"].append(log_loss(yt, p))
        meta_scores["brier"].append(brier_score_loss(yt, p))
        print(f"  Meta Fold {fold}: AUC={meta_scores['auc'][-1]:.4f}")

    return meta_scores
