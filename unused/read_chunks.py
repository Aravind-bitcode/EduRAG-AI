# Convert chunks (text) to vectors ie. numbers
import requests
import os
import json
import pandas as pd


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


jsons = os.listdir("jsons") # List all the jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    if not json_file.endswith(".json"):
        continue
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        chunks = json.load(f)
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

# Save all embeddings to JSON file for easy viewing and vector search
output_file = "embeddings.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(my_dicts, f, indent=4, ensure_ascii=False)

print(f"\nSaved all 6,175 embeddings to {output_file}")