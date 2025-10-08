# Finance Domain Credit Decision Classification

An end-to-end **Machine Learning project** focused on predicting **credit decision categories (P1–P4)** in the **finance domain**.  
The solution integrates **EDA, feature engineering, imbalance handling (SMOTE), model comparison, hyperparameter tuning**, and **deployment using Gradio**.

---

## Project Overview

Financial institutions evaluate loan or credit card applications based on multiple behavioral and credit metrics.  
This project automates that decision-making using **machine learning**, predicting the category of approval (`P1`, `P2`, `P3`, `P4`) based on applicant and account-level data.

**Goal:**  
> Build a robust and interpretable ML pipeline that classifies credit decisions with high accuracy and deploys the best model for real-time prediction.

---

## Key Features

- Comprehensive **Exploratory Data Analysis (EDA)** and statistical validation.
- **Feature engineering** using domain metrics such as enquiries, delinquency levels, and account ages.
- **SMOTE balancing** to handle class imbalance.
- Model comparison among **Random Forest**, **XGBoost**, and **Artificial Neural Network (ANN)**.
- **Hyperparameter tuning** with GridSearchCV for best performance.
- **Feature importance ranking** and insights for explainability.
- **Deployment demo** via Gradio web app for quick inference.

---

## Dataset Information

Three Excel files were used:

| File Name | Description |
|------------|--------------|
| `dataset1.xlsx` | Primary applicant-level dataset |
| `dataset2.xlsx` | Supplementary dataset for model features |
| `Description.xlsx` | Metadata and feature definitions used for documentation and app labels |

**Target Column:** `Approved_Flag`  
(Possible values: `P1`, `P2`, `P3`, `P4`)

---

## Exploratory Data Analysis

- Checked for **missing values**, **outliers**, and **data imbalance**.
- Conducted **Chi-Square tests** for categorical feature relevance.
- Computed **correlation matrices** and visualized relationships.
- Identified most influential variables for approval prediction.

---

##  Data Preprocessing

- **Encoding:** Label Encoding for categorical variables.  
- **Scaling:** StandardScaler & MinMaxScaler for numerical features.  
- **Balancing:** SMOTE to handle target imbalance.  
- **Splitting:** 80–20 train/test split using stratification.

---

## Model Development

### 1. Random Forest Classifier  
- Served as the baseline model.  
- Accuracy: **~77.1%**

### 2. XGBoost Classifier (Tuned)  
- Tuned via `GridSearchCV` for hyperparameters:
  ```python
  {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}
  ```

---
## Model Performance Summary

### Test Accuracy: ~78.8%

---

### Top Features
- `enq_L3m`  
- `Age_Oldest_TL`  
- `pct_PL_enq_L6m_of_ever`  
- `max_recent_level_of_deliq`  
- `num_std_12mts`  
- `time_since_recent_enq`  
- `max_deliq_12mts`  
- `recent_level_of_deliq`  
- `num_deliq_6_12mts`  
- `Age_Newest_TL`

---

### Artificial Neural Network (ANN)
- **Frameworks Used:** Keras / TensorFlow  
- **F1-Weighted Score:** ~0.76  
- **Purpose:** Served as a deep learning comparison baseline to evaluate classical ML vs. neural networks.

---

### Model Evaluation Metrics

| **Model**            | **Accuracy** | **Weighted F1** | **Remarks**                        |
|----------------------|--------------|-----------------|------------------------------------|
| Random Forest        | 0.7711       | 0.75            | Baseline                           |
| XGBoost (tuned)      | 0.7881       | 0.77            | 🏆 **Best model**                   |
| ANN (MLP)            | 0.758        | 0.76            | Stable but slightly lower          |

---

Confusion Matrices and Classification Reports were generated for all models.  
Further metrics like **ROC-AUC**, **Recall**, and **Precision** can be added for deeper insights.

---

## Folder Structure

Finance_Domain_Project/
│
├── Finance_Domain_Project.ipynb # Main analysis & training notebook
├── app.py # Gradio deployment script
├── dataset1.xlsx # Primary dataset (Part 1)
├── dataset2.xlsx # Secondary dataset (Part 2)
├── Description.xlsx # Feature description and metadata
├── xgboost_model.joblib # Saved best XGBoost model
├── scaler.joblib # Saved StandardScaler object
└── README.md # Project documentation

---

## Future Improvements

- Add **Stratified K-Fold Cross Validation** for robust generalization.  
- Include **ROC-AUC**, **PR-AUC**, and **per-class recall** metrics.  
- Move **SMOTE** inside the CV pipeline to prevent data leakage.  
- Integrate **SHAP explainability** for feature impact visualization.  
- Apply **probability calibration** using `CalibratedClassifierCV`.  
- Deploy full pipeline on **Streamlit / FastAPI** for advanced dashboards.

---

## Learning Outcomes

- Practical application of **Machine Learning** in the **Finance and Credit Risk** domain.  
- Experience with **data imbalance handling**, **model tuning**, **feature importance**, and **deployment**.  
- Clear understanding of **model interpretability** and **reproducibility** in production ML workflows.

---

## Results Snapshot

| **Metric**          | **Value**                     |
|----------------------|-------------------------------|
| Best Accuracy        | 78.81%                        |
| Best Model           | XGBoost (GridSearchCV)        |
| SMOTE Used           | Yes                         |
| Deployment           | Gradio Interface             |
| Domain               | Finance / Credit Decision Analytics |

---

##  Author

**Sourav Kumar**  
_Data Science & AI Enthusiast | ML Engineer | Data Analyst_  

**Email:** [souravmail003@gmail.com](mailto:souravmail003@gmail.com)  
**LinkedIn:** [linkedin.com/in/sourav-kumar-5814341b8](https://linkedin.com/in/sourav-kumar-5814341b8)

---

⭐ **If you found this project helpful, don’t forget to star the repository!**

  

