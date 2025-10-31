import os
import pandas as pd
from load_data import load_dataset
from base_models import get_svm_model, get_rf_model, get_mlp_model
from meta_model import generate_oof_predictions, train_meta_model
from evaluate import evaluate_models, summarize_results, run_statistical_test

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

# Caminhos base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("\n=== Iniciando Experimento Híbrido ===")

    # === 1. Carregamento da base ===
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Base carregada: {X_train.shape[0]} amostras, {X_train.shape[1]} features")

    # === 2. Modelos híbridos de referência ===
    pca = PCA(n_components=0.95, random_state=42)
    scaler = StandardScaler()

    models_dict = {
        "PCA+DT": Pipeline([
            ("scaler", scaler),
            ("pca", pca),
            ("dt", CalibratedClassifierCV(
                estimator=DecisionTreeClassifier(random_state=42),
                method="sigmoid", cv=3
            ))
        ]),
        "PCA+SVM": get_svm_model(),
        "PCA+RF": get_rf_model(),
        "PCA+MLP": get_mlp_model()
    }

    # === 3. Avaliação dos híbridos ===
    results = evaluate_models(X_train, y_train, models_dict)
    summary = summarize_results(results, save_path=os.path.join(RESULTS_DIR, "metrics.csv"))
    print("\n=== Resultados dos Modelos Híbridos ===")
    print(summary)

    # === 4. Sistema Híbrido Stacking (nível-1) ===
    print("\n=== Treinando Sistema Híbrido Stacking ===")
    oof_df, trained_models = generate_oof_predictions(X_train, y_train, models_dict)
    meta_scores = train_meta_model(oof_df, y_train)

    # Calcula médias das métricas
    stacking_results = {
        "Modelo": ["STACK_LR"],
        "ACC": [f"{sum(meta_scores['acc'])/len(meta_scores['acc']):.4f}"],
        "AUC": [f"{sum(meta_scores['auc'])/len(meta_scores['auc']):.4f}"],
        "LogLoss": [f"{sum(meta_scores['logloss'])/len(meta_scores['logloss']):.4f}"],
        "Brier": [f"{sum(meta_scores['brier'])/len(meta_scores['brier']):.4f}"]
    }

    df_stacking = pd.DataFrame(stacking_results)
    df_stacking.to_csv(os.path.join(RESULTS_DIR, "stacking_results.csv"), index=False)
    print("\n=== Resultados do Stacking ===")
    print(df_stacking)

    # Adiciona o stacking ao conjunto para comparação
    results["STACK_LR"] = meta_scores
    summary = summarize_results(results, save_path=os.path.join(RESULTS_DIR, "metrics_all.csv"))

    # === 5. Teste de Hipótese Geral (com stacking) ===
    print("\n=== Teste Estatístico Final ===")
    run_statistical_test(results, metric="acc", save_prefix=os.path.join(RESULTS_DIR, "autorank_final"))

    print("\n=== Experimento Finalizado com Sucesso ===")
    print(f"Arquivos salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
