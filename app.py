"""
EduRAG AI — RAG-Based Video Lecture Teaching Assistant
======================================================
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

PORT = int(os.environ.get("PORT", 5000))
EMBEDDINGS_FILE = "embeddings.joblib"

# Fallback dataset if embeddings file is missing or corrupted
DEFAULT_LECTURE_CHUNKS = [
    {"chunk_id": 1, "file_name": "01_intro.json", "title": "Installing VS Code & How Websites Work", "start": 0.0, "end": 120.0, "text": "VS Code is the industry standard code editor for modern Web Development. Websites work via client-server HTTP requests rendering HTML, CSS, and JavaScript in the browser.", "audio_file": "01_intro.mp3"},
    {"chunk_id": 2, "file_name": "02_html_structure.json", "title": "Basic Structure of an HTML Website", "start": 15.0, "end": 180.0, "text": "Every HTML5 document requires a <!DOCTYPE html> declaration, <html> root element, <head> for metadata/title, and <body> containing visible website content.", "audio_file": "02_html_structure.mp3"},
    {"chunk_id": 3, "file_name": "03_headings_links.json", "title": "Headings, Paragraphs and Links in HTML", "start": 30.0, "end": 210.0, "text": "Use h1 through h6 tags for semantic content hierarchy. Paragraphs use <p> tags, and anchor tags <a href='...'> create hyperlinks to internal and external web pages.", "audio_file": "03_headings_links.mp3"},
    {"chunk_id": 4, "file_name": "04_css_box_model.json", "title": "CSS Box Model: Margin, Padding & Borders", "start": 45.0, "end": 240.0, "text": "The CSS Box Model consists of content box, padding inside borders, border width, and margin spacing outside elements. Box-sizing border-box simplifies layout math.", "audio_file": "04_css_box_model.mp3"},
    {"chunk_id": 5, "file_name": "05_forms_inputs.json", "title": "Forms and Input Tags in HTML", "start": 20.0, "end": 190.0, "text": "HTML forms collect user data using <form action='...' method='POST'> with <input type='text'>, <input type='email'>, <input type='password'>, and <button type='submit'>.", "audio_file": "05_forms_inputs.mp3"},
    {"chunk_id": 6, "file_name": "06_inline_block.json", "title": "Inline vs Block Elements in HTML", "start": 10.0, "end": 160.0, "text": "Block-level elements (div, h1, p) take full width and start on new lines. Inline elements (span, a, img) wrap tightly around content without breaking lines.", "audio_file": "06_inline_block.mp3"},
    {"chunk_id": 7, "file_name": "07_media_tags.json", "title": "Video, Audio & Media in HTML", "start": 25.0, "end": 220.0, "text": "Embed responsive video and audio using <video controls src='...'> and <audio controls src='...'> tags with fallback sources and custom media styling.", "audio_file": "07_media_tags.mp3"},
    {"chunk_id": 8, "file_name": "08_semantic_html.json", "title": "Semantic Tags in HTML & SEO", "start": 35.0, "end": 250.0, "text": "Semantic HTML tags (header, nav, main, article, section, footer, aside) improve search engine optimization (SEO) and web accessibility for screen readers.", "audio_file": "08_semantic_html.mp3"},
]

df_embeddings = None


def load_embeddings():
    """Loads vector embeddings dataset into memory with robust fallback."""
    global df_embeddings
    try:
        if os.path.exists(EMBEDDINGS_FILE):
            print(f"Loading '{EMBEDDINGS_FILE}' into memory...")
            df_embeddings = joblib.load(EMBEDDINGS_FILE)
            print(f"Loaded {len(df_embeddings)} vector chunks successfully!")
            return
    except Exception as err:
        print(f"Error loading joblib file ({err}). Using fallback dataset.")

    try:
        if os.path.exists("embeddings.json"):
            with open("embeddings.json", encoding="utf-8") as f:
                data = json.load(f)
            df_embeddings = pd.DataFrame.from_records(data)
            print(f"Loaded {len(df_embeddings)} vector chunks from json!")
            return
    except Exception as err:
        print(f"Error loading json file ({err}).")

    print("Initializing fallback Web Development lecture database.")
    df_embeddings = pd.DataFrame.from_records(DEFAULT_LECTURE_CHUNKS)


def get_query_embedding(query_text, model_name="bge-m3"):
    """Generates 1024-dim embedding vector if local Ollama API is running."""
    url = "http://localhost:11434/api/embed"
    payload = {"model": model_name, "input": [query_text]}
    try:
        response = requests.post(url, json=payload, timeout=2)
        if response.status_code == 200:
            return response.json()["embeddings"][0]
    except Exception:
        pass
    return None


def resolve_audio_filename(file_name):
    """Maps transcript filename to MP3 audio filename safely."""
    if not file_name:
        return ""
    base_mp3 = file_name.replace('.json', '.mp3')
    try:
        if os.path.exists("audios"):
            if os.path.exists(os.path.join("audios", base_mp3)):
                return base_mp3
            if "_" in file_name:
                parts = file_name.split("_")
                if parts:
                    num_prefix = parts[0].lstrip("0")
                    for audio_f in os.listdir("audios"):
                        a_parts = audio_f.split("_")
                        if a_parts and a_parts[0].lstrip("0") == num_prefix:
                            return audio_f
    except Exception:
        pass
    return base_mp3


def search_similar_chunks(query_vector, query_text="", top_k=4):
    """Computes Cosine Similarity math between query and video lecture chunks safely."""
    global df_embeddings
    if df_embeddings is None or len(df_embeddings) == 0:
        df_embeddings = pd.DataFrame.from_records(DEFAULT_LECTURE_CHUNKS)

    try:
        if query_vector is not None and 'embedding' in df_embeddings.columns:
            chunk_embeddings = np.array(df_embeddings['embedding'].tolist())
            query_vector_2d = np.array([query_vector])
            similarities = cosine_similarity(query_vector_2d, chunk_embeddings)[0]
        else:
            texts = df_embeddings['text'].astype(str).tolist()
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts + [query_text])
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

        df_copy = df_embeddings.copy()
        df_copy['similarity'] = similarities
        top_df = df_copy.sort_values(by='similarity', ascending=False).head(top_k)

        results = []
        for _, row in top_df.iterrows():
            file_name = str(row.get('file_name', ''))
            title = str(row.get('title', ''))
            if not title:
                title = file_name.replace('.json', '').replace('.mp3', '')

            if " - Tutorial #" in title:
                clean_title = title.split(" - Tutorial #")[0]
            elif "_" in title:
                parts = title.split("_", 1)
                clean_title = parts[1] if len(parts) > 1 else title
            else:
                clean_title = title

            audio_file = resolve_audio_filename(file_name)

            results.append({
                "chunk_id": int(row.get('chunk_id', 1)),
                "file_name": file_name,
                "title": clean_title,
                "start": float(row.get('start', 0.0)),
                "end": float(row.get('end', 0.0)),
                "text": str(row.get('text', '')).strip(),
                "similarity": float(row.get('similarity', 0.85)),
                "audio_file": audio_file
            })
        return results
    except Exception as err:
        print(f"Error in search_similar_chunks: {err}")
        return DEFAULT_LECTURE_CHUNKS[:top_k]


def generate_llm_response(prompt, model_name="llama3"):
    """Queries Ollama LLM if available."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        res = requests.post(url, json=payload, timeout=3)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception:
        pass
    return None


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class RAGServerHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open("index.html", "rb") as f:
                self.wfile.write(f.read())
            return

        if parsed.path.startswith("/audios/"):
            audio_rel_path = urllib.parse.unquote(parsed.path[1:])
            if not os.path.exists(audio_rel_path):
                audio_filename = os.path.basename(audio_rel_path)
                resolved_name = resolve_audio_filename(audio_filename)
                audio_rel_path = os.path.join("audios", resolved_name)

            if os.path.exists(audio_rel_path):
                file_size = os.path.getsize(audio_rel_path)
                range_header = self.headers.get('Range')

                if range_header:
                    try:
                        byte_range = range_header.strip().split("=")[1]
                        if "-" in byte_range:
                            parts = byte_range.split("-", 1)
                            start_byte = int(parts[0]) if parts[0] else 0
                            end_byte = int(parts[1]) if parts[1] else file_size - 1
                        else:
                            start_byte = int(byte_range)
                            end_byte = file_size - 1
                    except Exception:
                        start_byte = 0
                        end_byte = file_size - 1

                    if start_byte >= file_size:
                        start_byte = file_size - 1
                    if end_byte >= file_size:
                        end_byte = file_size - 1

                    content_length = end_byte - start_byte + 1

                    self.send_response(206)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start_byte}-{end_byte}/{file_size}")
                    self.send_header("Content-Length", str(content_length))
                    self.end_headers()

                    with open(audio_rel_path, "rb") as f:
                        f.seek(start_byte)
                        self.wfile.write(f.read(content_length))
                    return
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(file_size))
                    self.end_headers()
                    with open(audio_rel_path, "rb") as f:
                        self.wfile.write(f.read())
                    return
            else:
                self.send_error(404, "Audio file not found")
                return

        if parsed.path == "/api/course_info":
            total = len(df_embeddings) if df_embeddings is not None else len(DEFAULT_LECTURE_CHUNKS)
            info = {
                "total_chunks": total,
                "embedding_model": "Ollama bge-m3 (1024-dim)",
                "topics": [
                    "Installing VS Code & How Websites Work",
                    "Your First HTML Website",
                    "Basic Structure of an HTML Website",
                    "Heading, Paragraphs and Links",
                    "Image, Lists, and Tables in HTML",
                    "SEO and Core Web Vitals in HTML",
                    "Forms and Input Tags in HTML",
                    "Inline & Block Elements in HTML",
                    "Id & Classes in HTML",
                    "Video, Audio & Media in HTML",
                    "Semantic Tags in HTML"
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(info).encode('utf-8'))
            return
            
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/search":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                query = data.get("query", "").strip()
                
                if not query:
                    self.send_error(400, "Empty query")
                    return
                
                query_vector = get_query_embedding(query)
                matches = search_similar_chunks(query_vector, query_text=query, top_k=4)
                
                context_lines = []
                for m in matches:
                    context_lines.append(f"- Video '{m['title']}' ({m['start']:.1f}s-{m['end']:.1f}s): {m['text']}")
                
                context_str = "\n".join(context_lines)
                prompt = f"""You are an AI Teaching Assistant for a Web Development video course.
Answer the user's question clearly using ONLY the provided course context below.
Cite the relevant video title and timestamps.

CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER:"""

                ai_response = generate_llm_response(prompt)
                if not ai_response:
                    if matches:
                        ai_response = f"Based on your search query **\"{query}\"**, the most relevant tutorial videos are listed below. Key context: {matches[0]['text']}"
                    else:
                        ai_response = f"Based on your search query **\"{query}\"**, relevant Web Development tutorial lecture segments have been retrieved below."

                response_data = {
                    "query": query,
                    "matches": matches,
                    "answer": ai_response
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                print(f"Error processing search: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        self.send_error(404, "Endpoint not found")


if __name__ == "__main__":
    load_embeddings()
    server_address = ('', PORT)
    httpd = ReusableTCPServer(server_address, RAGServerHandler)
    print(f"\n==========================================================")
    print(f"RAG AI Teaching Assistant Web Server Running on PORT {PORT}")
    print(f"==========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
