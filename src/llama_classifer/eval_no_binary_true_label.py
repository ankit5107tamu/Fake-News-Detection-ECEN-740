import pandas as pd

import sys

from sklearn.metrics import accuracy_score, classification_report



# === CLI argument ===

if len(sys.argv) != 2:

    print("Usage: python eval_binary_from_fine_label.py <predictions_csv>")

    sys.exit(1)



input_csv = sys.argv[1]



# === Mapping from fine-grained to binary labels ===

label_map = {

    "true": "true",

    "mostly-true": "true",

    "half-true": "false",

    "barely-true": "false",

    "pants-fire": "false",

    "false": "false"

}



# === Load and map ===

df = pd.read_csv(input_csv)

df = df.dropna(subset=["true_label", "predicted_label"])

df["true_label"] = df["true_label"].astype(str).str.lower()

df["binary_true_label"] = df["true_label"].map(label_map)

df["predicted_label"] = df["predicted_label"].astype(str).str.lower()



# === Evaluation ===

y_true = df["binary_true_label"]

y_pred = df["predicted_label"]



accuracy = accuracy_score(y_true, y_pred)

report = classification_report(

    y_true, y_pred,

    labels=["true", "false"],

    target_names=["true", "false"],

    zero_division=0

)



print(f"✅ Binary Accuracy (mapped from fine labels): {accuracy:.4f}")

print("=== Classification Report ===")

print(report)


