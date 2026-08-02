# Creating Chunks as Whisper model only processes .mp3 files

import whisper
import json 
import os

model = whisper.load_model("base")

audios = os.listdir("audios")

for audio in audios:
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number,title)

        result = model.transcribe(audio=f"audios/{audios}",language="hi",task="translate",fp16=False, word_timestamps=False)

        chunks=[]
        for segment in result["segments"]:
            chunks.append({"number": number, "title": title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})

        chunks_with_metadeta = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadeta,f)