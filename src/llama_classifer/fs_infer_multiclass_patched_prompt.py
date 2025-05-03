
import pandas as pd
import re
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Config ===
model_path = "../llama-3.1-8b-hf/"
input_csv = "pheme_merged_output.csv"  # should contain title, text, sentiment, emotion_score, label
output_csv = "llama_predictions_pheme_zero_shot_no_emo.csv"
mode = "zero-shot"  # or "zero-shot"
num_per_class = 1  # use 1 true + 1 false for binary classification

# === Load model ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# === Read dataset ===
df = pd.read_csv(input_csv, on_bad_lines='skip')
df = df.dropna(subset=['title', 'text', 'sentiment', 'emotion_score', 'label'])
df['label'] = df['label'].astype(str).str.lower()
valid_labels = ["true", "false"]
df = df[df['label'].isin(valid_labels)]

# === Balanced sampling: 1 true + 1 false
def get_balanced_binary_shots(df):
    shots = []
    for label in valid_labels:
        subset = df[df['label'] == label]
        if not subset.empty:
            shots.append(subset.sample(1))
    return pd.concat(shots).sample(frac=1).reset_index(drop=True)

shot_examples = get_balanced_binary_shots(df) if mode == "few-shot" else []
print("Few-shot class distribution:")
if mode == "few-shot":
    print(shot_examples['label'].value_counts())# if mode == "few-shot"
# === Prompt builder (diagnostic format)i
#Sentiment: {row['sentiment']}
#Emotion Score: {row['emotion_score']}
#This article has a sentiment of {row['sentiment']} and an emotion score of {row['emotion_score']}.

def build_prompt(title, text, sentiment, emotion_score):
    instruction = "You are a news classification assistant. Predict whether the news is TRUE or FALSE based on its content and emotional tone.\n\n"
    prompt = instruction
    if mode == "few-shot":
        for i, (_, row) in enumerate(shot_examples.iterrows(), 1):
            short_text = row['text'][:300].replace("\n", " ")
            prompt += f"""Example {i}:
Title: {row['title']}
Text: {short_text}
Label: {row['label'].upper()}

"""
        # Your article
        short_input_text = text[:300].replace("\n", " ")
        prompt += f"""Example {len(shot_examples) + 1}:
Title: {title}
Text: {short_input_text}
Label:"""
    else:
        short_input_text = text[:300].replace("\n", " ")
        prompt = instruction + f"""Title: {title}
Text: {short_input_text}
Choose one label: TRUE or FALSE
Answer:"""
    return prompt

# === Extract binary prediction
def extract_label_from_output(output):
    output = output.lower()
    match = re.search(r"\b(true|false)\b", output)
    return match.group(1) if match else "unknown"

# === Inference loop
predictions = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row['title'], row['text'], row['sentiment'], row['emotion_score'])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_label_from_output(decoded)

    predictions.append({
        "title": row['title'],
        "text": row['text'],
        "sentiment": row['sentiment'],
        "emotion_score": row['emotion_score'],
        "true_label": row['label'],
        "predicted_label": pred,
        "model_output": decoded
    })

    if idx % 10 == 0:
        print(f"[{idx}] Predicted: {pred} | True: {row['label']}")
        print("Prompt ===>")
        print(prompt[:400])
        print("Output ===>")
        print(decoded[:300])
        sys.stdout.flush()

# === Save results
pd.DataFrame(predictions).to_csv(output_csv, index=False)
print(f"✅ Predictions saved to {output_csv}")
