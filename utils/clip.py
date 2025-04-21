import os
import subprocess

def subclip(args):
    video_path, start_time, end_time, output_file = args
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(end_time - start_time),
        "-c", "copy",
        # "-c:v", "libx264",
        # "-preset", "ultrafast",
        # "-c:a", "aac",
        f"{output_file}"
    ])