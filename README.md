# 🎓 EduRAG AI — RAG-Based Video Lecture Teaching Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-bge--m3-orange?style=for-the-badge)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green?style=for-the-badge&logo=openai)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Cosine--Similarity-gold?style=for-the-badge&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**EduRAG AI** is a multimodal Retrieval-Augmented Generation (RAG) conversational assistant designed for online educational courses. It transcribes video lectures via **OpenAI Whisper**, indexes 1024-dimensional semantic text embeddings using **Ollama (bge-m3)**, and routes student questions directly to the exact timestamp answer inside the video playback player.

---

## ✨ Key Features

- 🎧 **OpenAI Whisper Audio Pipeline**: Transcribes raw video audio into timestamped JSON text chunks.
- 🧠 **1024D Semantic Vector Retrieval**: Encodes course concepts into 1024-dimensional vectors with `Ollama bge-m3` and computes cosine similarity for high-precision Q&A retrieval.
- 🎯 **Direct Timestamp Seeking**: Automatically navigates video audio players (HTTP 206 Partial Content range requests) to the exact second a topic is discussed.
- 💬 **Context-Aware LLM Synthesis**: Generates accurate, cited responses answering student questions exclusively from lecture context.

---

## 🛠️ Tech Stack

- **Backend Framework**: `Python`, `http.server`, `socketserver`
- **Transcription**: `OpenAI Whisper`, `ffmpeg`
- **Vector Embeddings & NLP**: `Ollama (bge-m3)`, `Joblib`, `Pandas`, `NumPy`
- **Similarity Search**: `Scikit-Learn (Cosine Similarity)`
- **LLM Generator**: `Ollama (llama3 / mistral)`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally (`ollama pull bge-m3`)

### Installation & Run

```bash
# 1. Clone repository
git clone https://github.com/Aravind-bitcode/EduRAG-AI.git
cd EduRAG-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch EduRAG Web Server
python app.py
```

Open `http://localhost:5000` in your browser to interact with the AI Teaching Assistant!

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — Copyright (c) Aravind Johindkumar.
