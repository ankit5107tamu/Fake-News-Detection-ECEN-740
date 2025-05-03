
import pandas as pd
import re
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# === Config ===
model_path = "../llama-3.1-8b-hf/"
input_csv = "liar_merged_output.csv"  # should contain title, text, sentiment, emotion_score, label
output_csv = "llama_predictions_multiclass_liar_with_emo.csv"
mode = "zero-shot"  # or "zero-shot"
num_per_class = 2  # for few-shot mode

# === Load model ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# === Read dataset ===
df = pd.read_csv(input_csv, on_bad_lines='skip')
df = df.dropna(subset=['title', 'text', 'sentiment', 'emotion_score', 'label'])
df['label'] = df['label'].astype(str).str.lower()


# === Valid class labels ===
valid_labels = ["true", "false", "barely-true", "mostly-true", "half-true", "pants-fire"]
df = df[df['label'].isin(valid_labels)]

# === Balanced few-shot selection ===
def get_balanced_multiclass_shots(df, num_per_class=1):
    shots = []
    for label in valid_labels:
        subset = df[df['label'] == label]
        if len(subset) >= num_per_class:
            shots.append(subset.sample(num_per_class))
    return pd.concat(shots).sample(frac=1).reset_index(drop=True)

shot_examples = get_balanced_multiclass_shots(df, num_per_class=num_per_class) if mode == "few-shot" else []
print("Few-shot class distribution:")
if mode == "few-shot":
    print(shot_examples['label'].value_counts()) #if mode == "few-shot"
# === Prompt builder with instruction header and shortened text ===
#Sentiment: {row['sentiment']}
#Emotion Score: {row['emotion_score']}
#This article has a sentiment of {row['sentiment']} and an emotion score of {row['emotion_score']}.

def build_prompt(title, text, sentiment, emotion_score):
    instruction = "You are a helpful assistant. Classify each article as one of the following: TRUE, FALSE, barely-true, mostly-true, half-true, or pants-fire.\n\n"
    if mode == "few-shot":
        prompt = instruction
        for _, row in shot_examples.iterrows():
            short_text = row['text'][:300].replace("\n", " ")
            prompt += f"""News Article:
Title: {row['title']}
Text: {short_text}
Sentiment: {row['sentiment']}
Emotion Score: {row['emotion_score']}
This article has a sentiment of {row['sentiment']} and an emotion score of {row['emotion_score']}.
Label: {row['label']}

"""
        short_input_text = text[:300].replace("\n", " ")
        prompt += f"""News Article:
Title: {title}
Text: {short_input_text}
Sentiment: {row['sentiment']}
Emotion Score: {row['emotion_score']}
This article has a sentiment of {row['sentiment']} and an emotion score of {row['emotion_score']}.
Label:"""
    else:
        short_input_text = text[:300].replace("\n", " ")
        prompt = instruction + f"""News Article:
Title: {title}
Text: {short_input_text}
Sentiment: {row['sentiment']}
Emotion Score: {row['emotion_score']}
This article has a sentiment of {row['sentiment']} and an emotion score of {row['emotion_score']}.

Which of the following best describes the article?
Options: TRUE, FALSE, barely-true, mostly-true, half-true, pants-fire
Answer:"""
    return prompt

# === Extract prediction label using regex ===
def extract_label_from_output(output):
    output = output.lower()
    match = re.search(r"(true|false|barely-true|mostly-true|half-true|pants-fire)", output)
    return match.group(0) if match else "unknown"

# === Inference loop ===
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
        print("Prompt Start ===>")
        print(prompt[:300])
        print("Generated Output ===>")
        print(decoded[:300])
        sys.stdout.flush()

# === Save predictions ===
pd.DataFrame(predictions).to_csv(output_csv, index=False)
print(f"✅ Predictions saved to {output_csv}")
