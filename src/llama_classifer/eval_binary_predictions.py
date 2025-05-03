
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, classification_report

# === CLI argument ===
if len(sys.argv) != 2:
    print("Usage: python eval_binary_predictions.py <predictions_csv>")
    sys.exit(1)

input_csv = sys.argv[1]

# === Read and normalize ===
df = pd.read_csv(input_csv)
df = df.dropna(subset=["binary_true_label", "predicted_label"])

y_true = df["binary_true_label"].astype(str).str.lower()
y_pred = df["predicted_label"].astype(str).str.lower()

# === Evaluation ===
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true, y_pred,
    target_names=["true", "false"],
    labels=["true", "false"],
    zero_division=0
)

print(f"✅ Binary Accuracy: {accuracy:.4f}")
print("=== Classification Report ===")
print(report)
