from load_data import load_dataset
from base_models import get_svm_model, get_rf_model, get_mlp_model
from meta_model import generate_oof_predictions, train_meta_model
import pandas as pd

def main():
    X_train, X_test, y_train, y_test = load_dataset()

    base_models = {
        "SVM": get_svm_model(),
        "RF": get_rf_model(),
        "MLP": get_mlp_model()
    }

    # Gera predições out-of-fold (OOF)
    oof_df, trained_models = generate_oof_predictions(X_train, y_train, base_models)

    print("\n=== Predições OOF (amostra) ===")
    print(oof_df.head())

    # Treina meta-aprendiz
    meta_scores = train_meta_model(oof_df, y_train)

    print("\n=== Desempenho do Meta-Aprendiz ===")
    for k, v in meta_scores.items():
        print(f"{k.upper()}: {sum(v)/len(v):.4f}")

if __name__ == "__main__":
    main()
