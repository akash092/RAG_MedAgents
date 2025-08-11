from Bio import Entrez
from tqdm import tqdm
import os
import time

# === CONFIG ===
EMAIL = "akash092@utexas.edu"  # Required by NCBI
OUTPUT_DIR = "pubmed/"
SEARCH_TERM = "hypertension"      # second most common disease(after pain) in our dataset
MAX_ARTICLES = 100                # Number of abstracts to fetch
SLEEP_TIME = 0.5                  # Be kind to NCBI API

# === SETUP ===
Entrez.email = EMAIL
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. SEARCH ===
print(f"Searching PubMed for: {SEARCH_TERM}")
search_handle = Entrez.esearch(db="pubmed", term=SEARCH_TERM, retmax=MAX_ARTICLES)
search_results = Entrez.read(search_handle)
search_handle.close()

pmids = search_results["IdList"]
print(f"Found {len(pmids)} articles.")

# === 2. FETCH + SAVE ===
for i, pmid in enumerate(tqdm(pmids, desc="Downloading abstracts")):
    try:
        fetch_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract_text = fetch_handle.read()
        fetch_handle.close()

        with open(os.path.join(OUTPUT_DIR, f"pubmed_{pmid}.txt"), "w") as f:
            f.write(abstract_text)

        time.sleep(SLEEP_TIME)  # Rate limit
    except Exception as e:
        print(f"Error fetching {pmid}: {e}")
        continue

print(f"Saved abstracts to: {OUTPUT_DIR}")