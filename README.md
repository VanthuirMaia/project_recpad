# 🧠 Sistema Híbrido de Reconhecimento de Padrões

## 🎯 Título

**Sistema Híbrido Baseado em PCA e Meta-Aprendiz para Reconhecimento de Padrões no Diagnóstico de Câncer de Mama**

---

## 🧩 1. Contextualização

Este projeto foi desenvolvido como parte da disciplina **Reconhecimento de Padrões** do **Programa de Pós-Graduação em Engenharia da Computação (PPGEC/UPE)**.  
O objetivo é implementar e avaliar um **sistema híbrido de classificação**, combinando técnicas de **extração de características (PCA)** com **modelos supervisionados** e um **meta-aprendiz de empilhamento (stacking)**.

A proposta atende aos requisitos de:

- Implementar um sistema híbrido;
- Compará-lo com outros híbridos da literatura;
- Aplicar testes estatísticos de hipótese (Friedman + Nemenyi);
- Reportar métricas e análises conforme metodologia científica.

---

## 📚 2. Estrutura do Projeto

```
project_recpad/
│
├── src/
│   ├── load_data.py             # Carregamento e divisão do dataset
│   ├── preprocessing.py         # Escalonamento, PCA, seleção de features
│   ├── base_models.py           # Modelos de nível 0 (SVM, RF, MLP)
│   ├── meta_model.py            # Meta-aprendiz (Logistic Regression)
│   ├── evaluate.py              # Validação cruzada, métricas, autorank
│   ├── experiment.py            # Execução completa e integração dos módulos
│   └── test_evaluate.py         # Execução isolada dos híbridos base
│
├── results/
│   ├── metrics.csv              # Métricas médias ± DP por modelo
│   ├── stacking_results.csv     # Desempenho do sistema stacking
│   ├── metrics_all.csv          # Comparativo geral (híbridos + stacking)
│   ├── autorank_final_report.txt
│   ├── autorank_final_plot.png
│   └── figures/
│
├── notebooks/                   # EDA e análises exploratórias
├── docs/                        # Artigo e slides da apresentação
├── requirements.txt
└── README.md
```

---

## 🧪 3. Metodologia

### **Etapa 1 — Pré-processamento**

- Escalonamento das variáveis (`StandardScaler`);
- Redução de dimensionalidade (`PCA`, 95% de variância retida);
- Alternativamente, seleção de atributos por `mutual_info_classif`.

### **Etapa 2 — Modelos de nível 0**

| Modelo             | Espaço de entrada           | Característica                 |
| :----------------- | :-------------------------- | :----------------------------- |
| SVM (RBF)          | Dados escalados             | Fronteiras não lineares        |
| Random Forest      | Dados transformados via PCA | Robustez e interpretabilidade  |
| MLP (32 neurônios) | Dados escalados             | Representação não linear suave |
| Decision Tree      | Baseline simples            | Comparativo inicial            |

### **Etapa 3 — Meta-Aprendiz (nível 1)**

- Combina probabilidades OOF dos modelos base;
- Logistic Regression como meta-classificador;
- Treinado em validação cruzada estratificada (k=10).

### **Etapa 4 — Avaliação e Estatística**

- Métricas: Acurácia, AUC, LogLoss, Brier Score;
- Validação cruzada estratificada (10 folds);
- Testes de hipótese: Friedman + Nemenyi (`autorank`);
- Resultados salvos em `/results` (CSV + PNG + TXT).

---

## ⚙️ 4. Execução do Projeto

### **Instalação**

```bash
git clone https://github.com/seuusuario/project_recpad.git
cd project_recpad
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### **Execução completa**

```bash
python -m src.experiment
```

### **Resultados gerados**

- `results/metrics.csv` → métricas médias dos modelos híbridos
- `results/stacking_results.csv` → desempenho do sistema híbrido stacking
- `results/autorank_final_report.txt` → teste estatístico completo
- `results/autorank_final_plot.png` → gráfico comparativo dos ranks

---

## 📈 5. Principais Resultados

| Modelo          | Acurácia Média | AUC Média   |
| :-------------- | :------------- | :---------- |
| PCA + DT        | ~0.94          | ~0.97       |
| PCA + RF        | ~0.95          | ~0.98       |
| PCA + SVM       | ~0.97          | ~0.99       |
| PCA + MLP       | ~0.98          | ~0.99       |
| **Stacking LR** | **~0.98+**     | **~0.995+** |

Resultados demonstram ganho de performance do sistema híbrido em relação aos métodos individuais.

---

## 🧾 6. Referências

- Pedregosa et al. (2011). _Scikit-learn: Machine Learning in Python_. JMLR.
- Demšar, J. (2006). _Statistical comparisons of classifiers over multiple data sets_. JMLR.
- Raschka, S. (2018). _ML Stack Ensemble Methods_.
- Dataset: _Breast Cancer Wisconsin (Diagnostic)_ — `sklearn.datasets.load_breast_cancer()`

---

## 👨‍💻 Autores

**Vanthuir Maia**  
**Luiz Vitor Póvoas**
