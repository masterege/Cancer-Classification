import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os

# Create output directory if it doesn't exist
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", filename="confusion_matrix.png"):
    """
    Plot and save a labeled confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path)
    print(f"[INFO] Saved confusion matrix to {save_path}")
    plt.show()

def plot_classification_report(report_dict, title="Classification Report", filename="classification_report.png"):
    """
    Visualize and save a classification report dictionary as a heatmap.
    """
    df = pd.DataFrame(report_dict).iloc[:-1, :].T  # Remove accuracy row
    plt.figure(figsize=(8, 4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path)
    print(f"[INFO] Saved classification report heatmap to {save_path}")
    plt.show()

def plot_model_accuracies(results, filename="model_accuracies.png"):
    """
    Plot and save model accuracy scores for comparison.
    """
    accuracies = {model: results[model]["accuracy"] for model in results}
    models = list(accuracies.keys())
    scores = list(accuracies.values())

    plt.figure(figsize=(8, 4))
    sns.barplot(x=scores, y=models, palette="viridis")
    plt.xlabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xlim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path)
    print(f"[INFO] Saved accuracy comparison to {save_path}")
    plt.show()
