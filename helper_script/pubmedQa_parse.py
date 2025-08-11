import json
import ast
import os

# === CONFIG ===
INPUT_FILE = "datasets/PubMedQA/test.txt"   # path to your source file
OUTPUT_DIR = "datasets/PubMedQA"        # folder for output files
OUTPUT_FILE = "test.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Skip the header (first line)
for i, line in enumerate(lines[1:], start=1):
    # Split by tab or multiple spaces (depends on file formatting)
    parts = line.strip().split("\t")
    if len(parts) < 4:
        # Try splitting by multiple spaces if tab fails
        parts = line.strip().split("  ")
        parts = [p for p in parts if p.strip()]  # remove empties

    if len(parts) < 4:
        print(f"Skipping malformed line {i}: {line}")
        continue

    question = parts[0].strip().strip('"')
    options_str = parts[1].strip()
    answer_idx = parts[2].strip()

    # Convert the options string to dict safely
    try:
        options = ast.literal_eval(options_str)
    except Exception as e:
        print(f"Error parsing options on line {i}: {e}")
        continue

    # Build JSON object
    item = {
        "question": question,
        "options": options,
        "answer_idx": answer_idx
    }

    # Save each question to a separate file
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, "a", encoding="utf-8") as out_f:
        json.dump(item, out_f)
        out_f.write("\n")

print(f"Done. Saved {len(os.listdir(OUTPUT_DIR))} JSON files to {OUTPUT_DIR}")
