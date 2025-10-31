from load_data import load_dataset
from base_models import get_svm_model, get_rf_model, get_mlp_model

def main():
    X_train, X_test, y_train, y_test = load_dataset()

    models = {
        "SVM": get_svm_model(),
        "RF": get_rf_model(),
        "MLP": get_mlp_model()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{name} -> Acurácia: {acc:.4f}")

if __name__ == "__main__":
    main()
