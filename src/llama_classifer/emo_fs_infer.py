import pandas as pd

from transformers import LlamaForCausalLM, AutoTokenizer

import torch

import re



# === Config ===

model_path = "../llama-3.1-8b-hf/"

input_csv = "fakenewsnet_clean_mod.csv"

output_csv = "predictions_llama.csv"

mode = "few-shot"  # "zero-shot" or "few-shot"

num_shots = 10       # for few-shot



# === Load Model ===

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")



# === Build Few-shot Examples (if needed) ===

df = pd.read_csv(input_csv)

real_shots = df[df['label'] == 'real'].sample(num_shots // 2)

fake_shots = df[df['label'] == 'fake'].sample(num_shots // 2)

shot_examples = pd.concat([real_shots, fake_shots]) if mode == "few-shot" else[]

#shot_examples = df.sample(num_shots) if mode == "few-shot" else []



def build_prompt(title, text):

    if mode == "few-shot":

        prompt = ""

        for _, row in shot_examples.iterrows():

            prompt += f"""News Article:

Title: {row['title']}

Text: {row['text']}

Sentiment: {row['sentiment']}

Emotion Score: {row['emotion_score']}

Label: {row['label']}



"""

        prompt += f"""News Article:

Title: {title}

Text: {text}

Sentiment: {sentiment}

Emotion Score: {emotion_score}

Label:"""



    else:  # zero-shot

        prompt = f"""News Article:

Title: {title}

Text: {text}

Sentiment: {sentiment}

Emotion Score: {emotion_score}                



Is this article real or fake?

Answer:"""

    return prompt



# === Run Inference ===

predictions = []

for idx, row in df.iterrows():

    prompt = build_prompt(row['title'], row['text'], row['sentiment'], row['emotion_score'])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=50)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)



    # Try to extract label

    match = re.search(r"(fake|real)", decoded.lower())

    pred = match.group(1) if match else "unknown"



    predictions.append({

        "title": row['title'],

        "text": row['text'],

        "true_label": row['label'],

        "predicted_label": pred,

        "model_output": decoded

    })
    if idx % 10 == 0:

        print(f"[{idx}] Predicted: {pred} | True: {row['label']}")

        print("Model Output:", decoded[:200], "...")  # optionally truncate

        import sys; sys.stdout.flush()
    
    if idx % 50 == 0 and idx > 0:

        pd.DataFrame(predictions).to_csv("predictions_partial.csv", index=False)






# === Save Output ===

pd.DataFrame(predictions).to_csv(output_csv, index=False)

print(f"✅ Predictions saved to {output_csv}")


