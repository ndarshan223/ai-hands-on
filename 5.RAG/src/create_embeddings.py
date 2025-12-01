import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
TEXT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed_texts"))
INDEX_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "embeddings", "faiss_index"))
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

print(f"Looking for text files in: {TEXT_DIR}")

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = []
metadata = []

def chunk_text(text, chunk_size=400):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

if not os.path.exists(TEXT_DIR) or not os.listdir(TEXT_DIR):
    print(f"⚠️  No text files found in {TEXT_DIR}")
    print("Please add .txt files to data/processed_texts/ or run extract_text.py first.")
    exit(1)

for file in os.listdir(TEXT_DIR):
    if file.endswith(".txt"):
        filepath = os.path.join(TEXT_DIR, file)
        print(f"Processing: {file}")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            for chunk in chunk_text(content):
                texts.append(chunk)
                metadata.append({"source": file})

if not texts:
    print("⚠️  No text content found to index.")
    exit(1)

print(f"Encoding {len(texts)} chunks...")
embeddings = model.encode(texts, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(INDEX_DIR, "cyber_index.faiss"))
np.save(os.path.join(INDEX_DIR, "texts.npy"), np.array(texts, dtype=object))
np.save(os.path.join(INDEX_DIR, "metadata.npy"), np.array(metadata, dtype=object))

print(f"✅ Created FAISS index with {len(texts)} chunks in {INDEX_DIR}")
