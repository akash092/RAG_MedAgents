import json
from collections import Counter
import spacy

nlp = spacy.load("en_ner_bc5cdr_md")

questions = []
# Load MedQA dataset
with open("datasets/MedQA/test.jsonl", "r") as f:
    for line in f:
        try:
            json_object = json.loads(line.strip())
            questions.append(json_object["question"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line.strip()} - {e}")

# Load PubMedQA dataset
with open("datasets/PubMedQA/test.jsonl", "r") as f:
    for line in f:
        try:
            json_object = json.loads(line.strip())
            questions.append(json_object["question"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line.strip()} - {e}")


disease_terms = []

for q in questions:
    doc = nlp(q)
    for ent in doc.ents:
        if ent.label_.lower() in ["disease", "condition", "problem", "disorder"]:
            disease_terms.append(ent.text.lower())

counts = Counter(disease_terms)
most_common = counts.most_common(20)

for disease, freq in most_common:
    print(f"{disease}: {freq}")

# Output captured for future reference
# pain: 349
# hypertension: 156
# fever: 133
# fatigue: 110
# tenderness: 104
# cough: 98
# shortness of breath: 87
# abdominal pain: 82
# swelling: 82
# diabetes mellitus: 60
# chest pain: 59
# diarrhea: 58
# rash: 57