import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

if len(sys.argv) != 2:
    print("Usage: python eval_accuracy_from_predictions.py <path_to_csv>")
    sys.exit(1)

input_csv = sys.argv[1]
# === Config ===
#input_csv = "llama_zero_rank_predictions.csv"

# === Load predictions ===
df = pd.read_csv(input_csv)
df = df.dropna(subset=["true_label", "predicted_label"])


# Normalize labels
y_true = df["true_label"].astype(str).str.lower()
y_pred = df["predicted_label"].astype(str).str.lower()

all_labels = sorted(list(unique_labels(y_true, y_pred)))
print("Labels:", all_labels)


# === Compute accuracy and report ===
accuracy = accuracy_score(y_true, y_pred)
#report = classification_report(y_true, y_pred, target_names=["true", "false"])
#report = classification_report(y_true, y_pred, target_names=["true", "false"], zero_division=0)
report = classification_report(y_true, y_pred, labels=all_labels, target_names=all_labels, zero_division=0)

print(f"✅ Accuracy: {accuracy:.4f}")
print("=== Classification Report ===")
print(report)
