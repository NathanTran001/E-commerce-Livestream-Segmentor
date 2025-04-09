import subprocess

import cv2
import threading
import time
import numpy as np
import pyaudio
import wave
import queue
import csv
from datetime import datetime
from moviepy.video.io.VideoFileClip import VideoFileClip
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from app import calc_landmark_list, pre_process_landmark, normalize_timestamps
from hand_gesture_recognizer.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# === CONFIGURABLE ===
TARGET_HAND_SIGN = "Peace"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
AUDIO_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
pose_duration = 1.5  # in seconds
frame_skip = 5

# === GLOBALS ===
audio_queue = queue.Queue()
recording = False
camera_index = 0
microphone_index = 0

# === Audio Recording Thread ===
def audio_recorder(filename, device_index):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

    frames = []
    print("[AUDIO] Recording started...")
    while recording:
        if stream.is_active():
            data = stream.read(CHUNK)
            frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("[AUDIO] Recording saved.")

# === Video Processing and Recording ===
def process_video_single_thread():
    global recording

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    keypoint_classifier = KeyPointClassifier()

    with open('hand_gesture_recognizer/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def new_writer(filename):
        return cv2.VideoWriter(filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    end_detected = False

    while True and not end_detected:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"recording_{timestamp}.avi"
        raw_video_filename = f"recording_{timestamp}.avi"
        audio_filename = f"recording_{timestamp}.wav"
        merged_filename = f"merged_{timestamp}.mp4"

        out = new_writer(video_filename)
        recording = True

        audio_thread = threading.Thread(target=audio_recorder, args=(audio_filename, microphone_index))
        audio_thread.start()

        print(f"[RECORDING] Started new recording: {video_filename}")

        timestamps_start = []
        timestamps_end = []
        frame_count = 0
        time_between_batches = 3  # seconds

        while recording:
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.flip(image, 1)

            out.write(image)

            if frame_count % frame_skip == 0:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)

                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        label = keypoint_classifier_labels[hand_sign_id]

                        timestamp = frame_count / frame_rate

                        if label == "Peace":
                            timestamps_start.append(timestamp)
                        elif label == "Open" and timestamps_start:
                            if not timestamps_end:
                                timestamps_end.append(timestamp)
                            else:
                                if abs(timestamps_end[0] - timestamp) < time_between_batches:
                                    timestamps_end.append(timestamp)
                                    if len(timestamps_end) >= 2 and (timestamps_end[-1] - timestamps_end[0]) >= pose_duration:
                                        end_detected = True
                                        recording = False
                                else:
                                    timestamps_end.clear()
                                    timestamps_end.append(timestamp)

            frame_count += 1

            cv2.imshow("Live", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False
                end_detected = True
                break

        out.release()
        recording = False
        audio_thread.join()

        merge_audio_video(raw_video_filename, audio_filename, merged_filename)

        timestamps_start = normalize_timestamps(timestamps_start)

        if not timestamps_start:
            print("No start points found")
            break

        segment_name = 1
        for idx, point_to_start in enumerate(timestamps_start):
            if len(timestamps_start) >= 2 and len(timestamps_start) > idx + 1:
                subclip(merged_filename, point_to_start, timestamps_start[idx + 1], f"{segment_name}.mp4")
                segment_name += 1

        start_time = timestamps_start[-1]
        if timestamps_end:
            end_time = timestamps_end[0]
            subclip(merged_filename, start_time, end_time, f"{segment_name}.mp4")
        else:
            subclip(merged_filename, start_time, VideoFileClip(merged_filename).duration, f"{segment_name}.mp4")

        if not end_detected:
            continue
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Live capture finished.")


def subclip(video_path, start_time, end_time, output_file):
    clip = VideoFileClip(video_path)
    subclip = clip.subclip(start_time, end_time)
    subclip.write_videofile(output_file, audio=True, codec="libx264", audio_codec="aac")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c", "copy",  # No re-encoding
        "-map", "0",  # Avoid duplicate streams
        f"{output_file}"
    ])


def merge_audio_video(video_path, audio_path, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ])


# === GUI: Camera + Microphone Selector ===
def list_camera_devices():
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append((i, f"Camera {i}"))
        cap.release()
    return available

def list_microphone_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append((i, info['name']))
    p.terminate()
    return devices


def start_gui():
    def on_start():
        global camera_index, microphone_index
        camera_index = cam_var.get()
        microphone_index = mic_var.get()
        root.destroy()
        process_video_single_thread()

    root = tk.Tk()
    root.title("Select Devices")

    cam_label = tk.Label(root, text="Camera:")
    cam_label.pack()
    cameras = list_camera_devices()
    cam_var = tk.IntVar(value=cameras[0][0])
    cam_menu = ttk.Combobox(root, values=[f"{name} (ID {i})" for i, name in cameras], state="readonly")
    cam_menu.current(0)
    cam_menu.pack()

    mic_label = tk.Label(root, text="Microphone:")
    mic_label.pack()
    microphones = list_microphone_devices()
    mic_var = tk.IntVar(value=microphones[0][0])
    mic_menu = ttk.Combobox(root, values=[f"{name} (ID {i})" for i, name in microphones], state="readonly")
    mic_menu.current(0)
    mic_menu.pack()

    def update_vars(*args):
        cam_var.set(cameras[cam_menu.current()][0])
        mic_var.set(microphones[mic_menu.current()][0])

    cam_menu.bind("<<ComboboxSelected>>", update_vars)
    mic_menu.bind("<<ComboboxSelected>>", update_vars)

    start_btn = tk.Button(root, text="Start Recording", command=on_start)
    start_btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
