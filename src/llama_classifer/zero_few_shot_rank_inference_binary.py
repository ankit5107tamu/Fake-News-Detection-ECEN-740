
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file with title, text, label, sentiment, emotion_score")
parser.add_argument("--output_csv", type=str, default="binary_rank_predictions.csv", help="Output predictions CSV")
parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
parser.add_argument("--mode", type=str, choices=["few-shot", "zero-shot"], default="few-shot", help="Choose inference mode")
args = parser.parse_args()

# === Binary label mapping ===
label_map = {
    "true": "true",
    "mostly-true": "true",
    "half-true": "false",
    "barely-true": "false",
    "pants-fire": "false",
    "false": "false"
}
fine_labels = list(label_map.keys())
binary_labels = ["true", "false"]

# === Load model/tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# === Load and prepare data ===
df = pd.read_csv(args.input_csv)
df = df.dropna(subset=["title", "text", "sentiment", "emotion_score", "label"])
df["label"] = df["label"].astype(str).str.lower()
df = df[df["label"].isin(fine_labels)]
df["binary_label"] = df["label"].map(label_map)

# === Few-shot prefix builder
def build_few_shot_prefix(few_shots_df):
    prefix = "You are a news classification assistant. Predict whether the article is TRUE or FALSE based on its content and emotional tone.\n\n"
    for i, row in enumerate(few_shots_df.itertuples(), 1):
        clean_text = row.text[:300].replace("\n", " ")
        label = label_map[row.label]
        prefix += f"""Example {i}:
Title: {row.title}
Text: {clean_text}
Sentiment: {row.sentiment}
Emotion Score: {row.emotion_score}
Label: {label.upper()}

"""
    return prefix

# === Prepare few-shot prefix if needed
few_shot_prefix = ""
if args.mode == "few-shot":
    few_shots = df.groupby("label").apply(lambda x: x.sample(n=2, random_state=42)).reset_index(drop=True)
    few_shot_prefix = build_few_shot_prefix(few_shots)

# === Prompt builder
def build_prompt(row):
    clean_text = row["text"][:300].replace("\n", " ")
    base = f"""Example:
Title: {row['title']}
Text: {clean_text}
Sentiment: {row['sentiment']}
Emotion Score: {row['emotion_score']}
Label:"""
    return few_shot_prefix + base if args.mode == "few-shot" else (
        "You are a news classification assistant. Predict whether the article is TRUE or FALSE.\n\n" + base
    )

# === Scoring
def score_candidate(prompt, candidate):
    full_input = prompt + " " + candidate.upper()
    inputs = tokenizer(full_input, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item()

# === Inference loop
results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row)
    score_true = score_candidate(prompt, "TRUE")
    score_false = score_candidate(prompt, "FALSE")
    prediction = "true" if score_true > score_false else "false"
    results.append({
        "title": row["title"],
        "text": row["text"],
        "sentiment": row["sentiment"],
        "emotion_score": row["emotion_score"],
        "fine_label": row["label"],
        "binary_true_label": row["binary_label"],
        "predicted_label": prediction,
        "score_true": score_true,
        "score_false": score_false
    })

# === Save
pd.DataFrame(results).to_csv(args.output_csv, index=False)
print(f"✅ Predictions saved to {args.output_csv}")
