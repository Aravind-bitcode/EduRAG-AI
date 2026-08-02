# Transcribes audio .mp3 files into timestamped JSON chunks using Whisper
import whisper
import json
import os

os.makedirs("jsons", exist_ok=True)

# Load whisper model ("base" for fast execution on CPU)
model = whisper.load_model("base")

audios = os.listdir("audios")

for audio in audios:
    if not audio.endswith(".mp3"):
        continue

    json_path = f"jsons/{audio}.json"
    if os.path.exists(json_path):
        print(f"Skipping {audio} (already transcribed)")
        continue

    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
    else:
        number = "0"
        title = os.path.splitext(audio)[0]

    print(f"Transcribing & Chunking: {audio}...")
    audio_file_path = os.path.join("audios", audio)

    result = model.transcribe(
        audio=audio_file_path,
        language="hi",
        task="translate",
        fp16=False,
        word_timestamps=False
    )

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, indent=4, ensure_ascii=False)

    print(f"Saved -> {json_path}")
