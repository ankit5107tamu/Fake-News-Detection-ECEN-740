import json

import re

import pandas as pd

import unicodedata



# Load the base CSV (with headers)

base_df = pd.read_csv("pheme_veracity_labels.csv")



# Load JSONL files

with open("predict_pheme_veracity_labels_senti.json") as f1, open("predict_pheme_veracity_labels_emo.json") as f2:

    sentiment_data = [json.loads(line) for line in f1 if line.strip()]

    emotion_data = [json.loads(line) for line in f2 if line.strip()]



# Normalize text: remove control chars, whitespace, trailing punctuation

def clean_text(s):

    s = unicodedata.normalize("NFKD", s)

    s = re.sub(r'[\r\n\t]+', ' ', s)

    s = re.sub(r'\s+', ' ', s)

    return s.strip().lower()



# Extract text & assistant value, trimming trailing task labels

def extract_text_and_assistant(output):

    text_match = re.search(r'Text:\s*(.*?)(?:Intensity|Label|Assistant):', output, re.DOTALL)

    assistant_match = re.search(r'Assistant:\n\s*(.*)', output, re.DOTALL)



    if text_match and assistant_match:

        raw_text = text_match.group(1).strip()

        # Remove trailing task labels like "intensity class:", "label:", etc.

        clean_sentence = re.sub(r'\s*(intensity class|intensity score|label)\s*[:]*\s*$', '', raw_text, flags=re.IGNORECASE)

        return clean_text(clean_sentence), assistant_match.group(1).strip()

    return None, None



# Build maps from cleaned JSONL

def parse_json(json_list):

    result = {}

    for entry in json_list:

        output = entry["output"]

        text, assistant = extract_text_and_assistant(output)

        if text and assistant:

            result[text] = assistant

    return result



# Build sentiment and emotion maps

sentiment_map = parse_json(sentiment_data)

emotion_map = parse_json(emotion_data)



# Match statements to maps

def match_row(row):

    stmt_clean = clean_text(row["statement"])

    row["sentiment"] = sentiment_map.get(stmt_clean, "")

    row["emotion_score"] = emotion_map.get(stmt_clean, "")

    return row



# Apply matching

base_df = base_df.apply(match_row, axis=1)



# Save merged result

base_df.to_csv("merged_output_pheme.csv", index=False)

print("✅ Output saved to 'merged_output_pheme.csv'")


