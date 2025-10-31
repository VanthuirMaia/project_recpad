import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def scale_data(X_train, X_test):
    """
    Padroniza os dados com média 0 e desvio padrão 1.
    Retorna DataFrames escalados mantendo os nomes das colunas.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train_scaled, X_test_scaled, variance_retained=0.95, random_state=42):
    """
    Aplica PCA para reduzir a dimensionalidade mantendo uma fração da variância explicada.
    Retorna os novos DataFrames transformados e o objeto PCA.
    """
    pca = PCA(n_components=variance_retained, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    cols = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
    X_train_pca = pd.DataFrame(X_train_pca, columns=cols, index=X_train_scaled.index)
    X_test_pca = pd.DataFrame(X_test_pca, columns=cols, index=X_test_scaled.index)

    return X_train_pca, X_test_pca, pca


def select_features(X_train_scaled, X_test_scaled, y_train, k=15):
    """
    Seleciona as k melhores features com base em informação mútua.
    Útil para comparar com o PCA em variações do sistema.
    """
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    selected_cols = X_train_scaled.columns[selector.get_support()]
    X_train_sel = pd.DataFrame(X_train_sel, columns=selected_cols, index=X_train_scaled.index)
    X_test_sel = pd.DataFrame(X_test_sel, columns=selected_cols, index=X_test_scaled.index)

    return X_train_sel, X_test_sel, selector
