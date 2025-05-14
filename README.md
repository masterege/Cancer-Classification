# 🧬 Cancer Severity Classification (2015–2024)

This project builds a machine learning pipeline to classify cancer severity based on global patient data from 2015 to 2024. The goal is to predict a binned severity class (Low, Medium, High) using medical and lifestyle factors.

---

## 📁 Dataset

- **Source**: [Kaggle Dataset by zahidmughal2343](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024)
- **Size**: 50,000 patients
- **Features**:
  - Age, Gender, Country
  - Genetic Risk, Air Pollution, Smoking, Alcohol Use, Obesity Level
  - Cancer Type & Stage
  - Treatment Cost, Survival Years
  - **Target**: `Target_Severity_Score` (binned into 3 classes)

---

## 🧪 ML Pipeline

- **Preprocessing**:
  - `OneHotEncoder` for categorical features
  - Binning `Target_Severity_Score` into 3 severity classes
  - Train/test split (80/20)
  - Feature scaling for numeric variables

- **Classifiers**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
  - XGBoost

- **Outputs**:
  - Accuracy comparison plot
  - Confusion matrix and classification report for each model
  - JSON metrics per model

---

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

---

## 📈 Sample Output

- `outputs/figures/`: confusion matrices, heatmaps, model comparison
- `outputs/reports/`: JSON files with classification reports per model

---

## 🛠 Requirements

- Python 3.8+
- scikit-learn
- pandas
- matplotlib
- seaborn
- xgboost

---

## 👤 Author

**Ege Ebiller**  
MSc Industrial Analytics, Uppsala University

---
