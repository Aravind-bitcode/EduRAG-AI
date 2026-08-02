# Convert JSON text chunks into vector embeddings via Ollama bge-m3 and save DataFrame using joblib
import requests
import os
import json
import pandas as pd
import joblib


def create_embedding(text_list, batch_size=32):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        r = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": batch})
        if r.status_code == 200:
            all_embeddings.extend(r.json()["embeddings"])
        else:
            raise Exception(f"Ollama API Error {r.status_code}: {r.text}")
    return all_embeddings


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join("jsons", json_file)
    with open(json_path, encoding="utf-8") as f:
        content = json.load(f)

    # Handle both format types (dict with "chunks" key or list directly)
    if isinstance(content, dict) and "chunks" in content:
        chunks = content["chunks"]
    elif isinstance(content, list):
        chunks = content
    else:
        continue

    print(f"Creating Embeddings for {json_file} ({len(chunks)} chunks)...")
    embeddings = create_embedding([c['text'] for c in chunks])

    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = chunk_id
        chunk['file_name'] = json_file
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
print(df.head())
print(f"\nTotal Chunks Processed: {len(df)}")

# Save DataFrame as embeddings.joblib file
output_file = "embeddings.joblib"
joblib.dump(df, output_file)
print(f"Saved DataFrame to '{output_file}' successfully!")

# Also save embeddings.json for reference/inspection
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(my_dicts, f, indent=4, ensure_ascii=False)
