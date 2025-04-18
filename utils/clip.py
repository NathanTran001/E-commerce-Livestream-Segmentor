import os
import subprocess


def subclip(args):
    video_path, start_time, end_time, output_file = args
    print(start_time)
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(end_time - start_time),
        "-c:v", "copy",
        "-c:a", "copy",
        "-avoid_negative_ts", "disabled",
        f"{output_file}"
    ])

