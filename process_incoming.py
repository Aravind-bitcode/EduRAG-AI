# Process incoming user queries: embed query, perform Cosine Similarity search, and generate LLM response
import os
import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_query_embedding(query_text, model_name="bge-m3"):
    """Generates embedding vector for user query via Ollama."""
    url = "http://localhost:11434/api/embed"
    payload = {"model": model_name, "input": [query_text]}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["embeddings"][0]
    else:
        raise Exception(f"Failed to generate embedding: {response.text}")


def search_similar_chunks(df, query_vector, top_k=5):
    """Calculates Cosine Similarity between query vector and all chunk embeddings."""
    chunk_embeddings = np.array(df['embedding'].tolist())
    query_vector_2d = np.array([query_vector])
    
    similarities = cosine_similarity(query_vector_2d, chunk_embeddings)[0]
    
    df_copy = df.copy()
    df_copy['similarity'] = similarities
    top_results = df_copy.sort_values(by='similarity', ascending=False).head(top_k)
    return top_results


def generate_llm_response(prompt, model_name="llama3"):
    """Sends context and query prompt to Ollama LLM endpoint."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"(LLM Model '{model_name}' not available or returned error. Pull a model via `ollama pull llama3` to enable LLM generation.)"
    except Exception as e:
        return f"(Could not connect to Ollama LLM: {e})"


def main():
    joblib_path = "embeddings.joblib"
    if not os.path.exists(joblib_path):
        print(f"Error: '{joblib_path}' not found! Run 'python process_json.py' first.")
        return

    print("Loading embeddings from embeddings.joblib...")
    df = joblib.load(joblib_path)
    print(f"Loaded {len(df)} chunk embeddings successfully!\n")

    while True:
        query = input("\nAsk your RAG Teaching Assistant (or type 'exit' to quit): ").strip()
        if not query or query.lower() == 'exit':
            break

        print(f"\nSearching relevant tutorial video chunks for: '{query}'...")
        query_vec = get_query_embedding(query)
        top_matches = search_similar_chunks(df, query_vec, top_k=3)

        print("\n" + "="*70)
        print("TOP MATCHING TUTORIAL VIDEO CHUNKS:")
        print("="*70)

        context_blocks = []
        for idx, row in top_matches.iterrows():
            title = row.get('title', row.get('file_name', 'Tutorial Video'))
            start_time = row.get('start', 0.0)
            end_time = row.get('end', 0.0)
            text = row.get('text', '')
            score = row['similarity']

            print(f"• Video/File: {title}")
            print(f"  Timestamp: {start_time:.2f}s - {end_time:.2f}s | Similarity Score: {score:.4f}")
            print(f"  Content: {text}")
            print("-" * 70)

            context_blocks.append(f"- Video: {title} (Timestamp {start_time:.1f}s-{end_time:.1f}s): {text}")

        # Construct RAG Prompt
        context_str = "\n".join(context_blocks)
        prompt = f"""You are an AI Teaching Assistant for a Web Development video course.
Answer the user's question using ONLY the provided course video context below.
Provide a clear, helpful response and cite the video title and timestamps where appropriate.

CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER:"""

        print("\nAI TEACHING ASSISTANT RESPONSE:")
        print("-" * 70)
        llm_reply = generate_llm_response(prompt, model_name="llama3")
        print(llm_reply)
        print("="*70)


if __name__ == "__main__":
    main()
