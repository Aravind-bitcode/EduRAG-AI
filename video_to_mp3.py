# Converts video files in "videos" folder to .mp3 audio in "audios" folder
import os
import subprocess

os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")
for file in files:
    if not (file.endswith(".mp4") or file.endswith(".webm") or file.endswith(".mkv")):
        continue
        
    if " - Tutorial #" in file:
        tutorial_number = file.split(" - Tutorial #")[1].split(" - ")[0]
        file_name = file.split(" Sigma")[0].strip()
    elif "Turorial #" in file:
        tutorial_number = file.split(" [")[0].split(" #")[1]
        file_name = file.split("|")[0].strip()
    else:
        tutorial_number = "0"
        file_name = os.path.splitext(file)[0]
    
    input_path = os.path.join("videos", file)
    output_path = os.path.join("audios", f"{tutorial_number}_{file_name}.mp3")
    
    if os.path.exists(output_path):
        print(f"Skipping {file} (already converted)")
        continue

    print(f"Converting {file} -> {output_path}...")
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-vn", output_path])
