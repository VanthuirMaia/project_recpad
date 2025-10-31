import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from autorank import autorank, create_report, plot_stats

# Caminho absoluto até a pasta de resultados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

RANDOM_STATE = 42


# =====================================================
# AVALIAÇÃO DOS MODELOS (VALIDAÇÃO CRUZADA)
# =====================================================
def evaluate_models(X, y, models_dict, n_splits=10):
    """
    Executa validação cruzada (Stratified KFold) para cada modelo em models_dict.
    Retorna um dicionário com as métricas de desempenho.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"acc": [], "auc": [], "logloss": [], "brier": []} for name in models_dict}

    for name, model in models_dict.items():
        print(f"\nTreinando e avaliando modelo: {name}")
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
            m = model
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = m.predict_proba(X.iloc[te_idx])[:, 1]
            yt = y.iloc[te_idx]

            results[name]["acc"].append(accuracy_score(yt, (p >= 0.5).astype(int)))
            results[name]["auc"].append(roc_auc_score(yt, p))
            results[name]["logloss"].append(log_loss(yt, p))
            results[name]["brier"].append(brier_score_loss(yt, p))
            print(f"  Fold {fold}: AUC={results[name]['auc'][-1]:.4f}")

    return results


# =====================================================
# RESUMO DAS MÉTRICAS
# =====================================================
def summarize_results(results, save_path=None):
    """
    Constrói tabela de métricas médias ± desvio padrão.
    """
    summary = []
    for model_name, metrics in results.items():
        summary.append({
            "Modelo": model_name,
            "ACC": f"{np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}",
            "AUC": f"{np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}",
            "LogLoss": f"{np.mean(metrics['logloss']):.4f} ± {np.std(metrics['logloss']):.4f}",
            "Brier": f"{np.mean(metrics['brier']):.4f} ± {np.std(metrics['brier']):.4f}",
        })
    df_summary = pd.DataFrame(summary)

    if save_path:
        df_summary.to_csv(save_path, index=False)
        print(f"\nResumo salvo em: {save_path}")

    return df_summary


# =====================================================
# TESTE DE HIPÓTESE (AUTORANK)
# =====================================================
def run_statistical_test(results, metric="acc", alpha=0.05, save_prefix=None):
    """
    Executa teste de hipótese (Friedman + Nemenyi) e salva o gráfico de forma compatível
    com todas as versões do autorank e matplotlib.
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame({k: v[metric] for k, v in results.items()})
    print("\n=== Teste de Hipótese (Autorank) ===")

    # Executa o teste estatístico
    res = autorank(df, alpha=alpha, verbose=False)

    # Gera o relatório textual
    try:
        report = create_report(res)
        if not isinstance(report, str):
            report = str(res)
    except Exception:
        print("⚠️ create_report falhou — gerando relatório básico.")
        report = str(res)

    print(report)

    # Caminhos de saída
    if save_prefix:
        report_path = f"{save_prefix}_report.txt"
        plot_path = f"{save_prefix}_plot.png"

        # Salva relatório textual
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Gera e salva gráfico via matplotlib
        try:
            plot_stats(res)               # exibe o gráfico
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()                   # fecha a figura
            print(f"\nGráfico salvo em: {plot_path}")
        except Exception as e:
            print(f"⚠️ Erro ao salvar gráfico: {e}")

        print(f"Relatório salvo em: {report_path}")

    return res
