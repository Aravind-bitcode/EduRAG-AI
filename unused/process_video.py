# Converts the video to .mp3
import os
import subprocess
os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")
for file in files:
    tutorial_number = file.split(" - Tutorial #")[1].split(" - ")[0]
    file_name = file.split(" Sigma")[0].strip()
    
    input_path = os.path.join("videos", file)
    output_path = os.path.join("audios", f"{tutorial_number}_{file_name}.mp3")
    
    print(f"Converting {file} -> {output_path}...")
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-vn", output_path])