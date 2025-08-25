# Credit Default Prediction Case Study

## 📌 Project Overview
This project is a technical case study simulating a real-world scenario in a major bank.  
The goal is to develop a **predictive model** to estimate the probability of credit default in real time and design a **credit policy** based on these predictions.

Deliverables include:
1. A predictive model to estimate default probability.
2. A credit policy recommendation (approval, rejection, or conditional approval).
3. A reproducible codebase.
4. A CSV file with predicted probabilities for new contracts.

---

## 📂 Data Sources
The project uses four main datasets:

- **base_cadastral.parquet** → Demographic and socioeconomic client data.
- **historico_emprestimos.parquet** → Past loan history (values, contract types, final status, etc.).
- **historico_parcelas.parquet** → Installment payments (amounts, due dates, delays).
- **base_submissao.parquet** → Current credit applications to score.

All datasets are linked via:
- `id_cliente` (client ID).
- `id_contrato` (contract ID).

---

## 🛠 Project Workflow

### 1. Problem Understanding
- Build a real-time default risk prediction model.
- Propose a business-oriented credit policy.

### 2. Target Variable Definition
Default can be defined in multiple ways:
- **FPD**: First Payment Default.
- **EVER30MOB03**: >30 days late within first 3 months.
- **OVER60MOB06**: >60 days cumulative delay within 6 months.

👉 The chosen definition must balance predictive power, interpretability, and business context.

### 3. Data Preparation
**Integration**
- Join client, loan, and installment tables into a modeling dataset.  

**Cleaning**
- Handle missing values (median imputation, category `Unknown`).  
- Remove duplicates and inconsistent records.  

**Outlier Detection**
- Boxplots and z-scores for numerical variables (age, income, loan value).  
- Treat extreme values using winsorization or log-transform.  

### 4. Exploratory Data Analysis (EDA)
- Univariate distributions (age, income, loan amount).  
- Bivariate analysis (default vs non-default).  
- Correlation matrix for numeric features.  
- Visualizations: histograms, boxplots, heatmaps.  

### 5. Feature Engineering
- **From loan history**: number of past loans, share of defaults, average loan value, client relationship duration.  
- **From installments**: mean delay days, share of overdue installments, recurrence of delinquency.  
- **From demographics**: age categories, income-to-loan ratio.  

### 6. Modeling
- **Baseline**: Logistic Regression (interpretability).  
- **Tree-based models**: LightGBM, XGBoost, Random Forest (performance).  
- **Validation**: Stratified K-Fold cross-validation.  
- **Metrics**: AUC-ROC, KS, Precision-Recall (recall prioritized to minimize risk exposure).  

### 7. Model Interpretation
- Feature importance with SHAP values.  
- Business insights: client profiles most at risk.  

### 8. Credit Policy
Cutoffs based on predicted probabilities:  
- `p < 0.2`: Approve.  
- `0.2 ≤ p < 0.5`: Conditional approval (reduced amount, higher interest).  
- `p ≥ 0.5`: Reject.  

---

