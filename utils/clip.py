import os
import subprocess


def subclip(args):
    date_time, video_path, start_time, end_time, output_file = args
    folder_path = f"videos/{date_time}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(start_time)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-t", str(end_time - start_time),
        "-c:v", "copy",  # Stream copy video - no re-encoding
        "-c:a", "copy",  # Stream copy audio - no re-encoding
        "-avoid_negative_ts", "make_zero",  # Helps with frame accuracy
        f"{folder_path}/{output_file}"
    ])

