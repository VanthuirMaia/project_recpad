import os
from load_data import load_dataset
from base_models import get_svm_model, get_rf_model, get_mlp_model
from evaluate import evaluate_models, summarize_results, run_statistical_test

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV

# === Diretórios base e de resultados ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)


def main():
    # === Carrega dataset ===
    X_train, X_test, y_train, y_test = load_dataset()

    # === Modelos híbridos de referência ===
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

    # === Avaliação dos modelos ===
    results = evaluate_models(X_train, y_train, models_dict)

    # === Exibe resumo e salva CSV ===
    summary = summarize_results(results, save_path=os.path.join(RESULTS_DIR, "metrics.csv"))
    print("\n=== Resumo das métricas ===")
    print(summary)

    # === Teste estatístico (Friedman + Nemenyi) ===
    run_statistical_test(results, metric="acc", save_prefix=os.path.join(RESULTS_DIR, "autorank"))


if __name__ == "__main__":
    main()
