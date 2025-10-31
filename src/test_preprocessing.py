from load_data import load_dataset
from preprocessing import scale_data, apply_pca, select_features

# Carrega os dados
X_train, X_test, y_train, y_test = load_dataset()

# Normaliza
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
print(f"Shape original: {X_train.shape}")
print(f"Shape escalado: {X_train_scaled.shape}")

# PCA
X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
print(f"PCA -> {X_train_pca.shape[1]} componentes mantiveram 95% da variância.")

# Seleção de features
X_train_sel, X_test_sel, selector = select_features(X_train_scaled, X_test_scaled, y_train, k=10)
print(f"Selecionadas {X_train_sel.shape[1]} melhores variáveis.")

