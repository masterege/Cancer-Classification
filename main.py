import pandas as pd
import json
import os
from src.data_loader import load_cancer_data, show_basic_info
from src.preprocessing import preprocess_data
from src.model import train_models
from src.utils import (
    plot_confusion_matrix,
    plot_classification_report,
    plot_model_accuracies
)

# Constants
DATA_PATH = "data/global_cancer_patients_2015_2024.csv"
RESULTS_DIR = "outputs/reports"
FIGURES_DIR = "outputs/figures"
TARGET_CLASS = "Severity_Class"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load and preview data
df = load_cancer_data(DATA_PATH)
show_basic_info(df)

# 2. Preprocess and split the data
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# 3. Train models
models, metrics = train_models(X_train, X_test, y_train, y_test, preprocessor)

# 4. Save results and plots
for name in models:
    print(f"[INFO] Saving outputs for {name}")
    y_pred = models[name].predict(X_test)

    # Save confusion matrix
    plot_confusion_matrix(y_test, y_pred, title=f"{name} - Confusion Matrix", filename=f"{name}_confusion.png")

    # Save classification report
    plot_classification_report(metrics[name]["report"], title=f"{name} - Classification Report", filename=f"{name}_report.png")

    # Save JSON metrics
    json_path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').lower()}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics[name], f, indent=4)
    print(f"[INFO] Saved metrics to {json_path}")

# 5. Save accuracy comparison plot
plot_model_accuracies(metrics, filename="all_model_accuracies.png")
