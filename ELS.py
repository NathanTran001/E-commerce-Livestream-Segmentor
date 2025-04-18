#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import argparse
import ctypes
import itertools
import os
import subprocess
import sys
from collections import Counter
from collections import deque
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import multiprocessing as mtp

from moviepy import VideoFileClip

from gui.GUI import VideoSplitterApp, start_sign, end_sign
from hand_gesture_recognizer.utils.cvfpscalc import CvFpsCalc
from hand_gesture_recognizer.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from hand_gesture_recognizer.model.point_history_classifier.point_history_classifier import PointHistoryClassifier

import threading
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from tkinterdnd2 import *

from utils.clip import subclip
from utils.hand_gesture import process_segment_with_hand
from utils.sign_detector import SignDetector, process_segment_with_sign
from utils.timestamps import normalize_timestamps, calculate_segment_boundaries

pose_duration = 0.8
time_between_batches = 0.4

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # For Windows
except:
    pass  # On non-Windows platforms

################################# GUI #################################


# Function to switch between frames (screens)
def run_main_in_thread(app):
    thread = threading.Thread(target=main, args=(app,))
    thread.daemon = True  # Ensures thread exits when main program ends
    thread.start()


################################# GUI #################################

def initialize_dynamic_parameters(fps):
    """
    Initialize global parameters based on frame rate.

    Args:
        fps (float): Frames per second of the video
    """
    global pose_duration, time_between_batches

    # Reference values optimized for 30fps video
    reference_fps = 30.0
    reference_pose_duration = pose_duration
    reference_time_between_batches = time_between_batches

    # Scale parameters linearly with frame rate ratio
    fps_ratio = fps / reference_fps

    # Apply scaling - slower videos need more time, faster videos need less
    pose_duration = reference_pose_duration * fps_ratio
    time_between_batches = reference_time_between_batches * fps_ratio

    print(f"- Pose duration: {pose_duration:.2f}s")
    print(f"- Time between batches: {time_between_batches:.2f}s")


def calculate_frames_to_skip(fps):
    """
    Calculate how many frames to skip while maintaining reliable detection.

    Args:
        fps (float): Frames per second of the video

    Returns:
        int: Number of frames to skip between processed frames
    """
    # Reference values for 30fps
    reference_fps = 30.0
    reference_min_detection_window = pose_duration
    reference_min_samples = 3

    # Scale detection window based on fps ratio
    fps_ratio = reference_fps / fps
    min_detection_window_seconds = reference_min_detection_window * fps_ratio

    # Calculate how many frames we need to check within our minimum detection window
    frames_in_window = fps * min_detection_window_seconds

    # Calculate maximum frames to skip while still getting enough samples
    max_skip = int(frames_in_window / reference_min_samples)

    print(f"- Min detection window: {min_detection_window_seconds:.2f}s")

    # Ensure we don't return 0 (which would mean process every frame)
    return max(1, max_skip - 1)


def process_video_parallel(video_path, num_segments, start_sign, end_sign):
    """
    Process a video by dividing it into segments and processing them in parallel.

    Args:
        video_path (str): Path to the video file
        num_segments (int): Number of segments to divide the video into

    Returns:
        tuple: Combined lists of start and end timestamps
    """
    # Calculate segment boundaries
    segments = calculate_segment_boundaries(video_path, num_segments)
    print(f"Dividing video into {num_segments} segments:")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i + 1}: {start:.2f}s - {end:.2f}s")

    # Create a pool of worker processes
    pool = mtp.Pool(processes=min(mtp.cpu_count(), num_segments))

    # Create a partial function with the video_path already set
    process_segment = partial(process_segment_with_sign, video_path=video_path, start_sign=start_sign, end_sign=end_sign)

    # Process all segments in parallel
    results = pool.starmap(process_segment, segments)

    # Close the pool
    pool.close()
    pool.join()

    # Combine results from all segments
    all_starts = []
    all_ends = []

    for starts, ends in results:
        all_starts.extend(starts)
        all_ends.extend(ends)

    # Sort the combined timestamps
    all_starts.sort()
    all_ends.sort()

    # Normalize timestamps if needed
    all_starts = normalize_timestamps(all_starts, time_between_batches, pose_duration)
    print(f"process_video_parallel: End before normalize: {all_ends}")
    all_ends = normalize_timestamps(all_ends, time_between_batches, pose_duration)
    print(f"process_video_parallel: End after normalize: {all_ends}")
    if len(all_ends) > 0 and len(all_starts) > 0:
        if all_ends[-1] > all_starts[0]:
            real_end = all_ends[-1]
            all_ends.clear()
            all_ends.append(real_end)
        else:
            all_ends.clear()
    else:
        all_ends.clear()
    print(f"process_video_parallel: End after cleaning: {all_ends}")

    return all_starts, all_ends


def main(app):
    start_time = time.perf_counter()

    # APP STARTS #################################################

    # Create App instance and run it

    video_path = app.selected_file.get()
    print(f"Processing video: {video_path}")

    # Get available CPU cores (or use a reasonable default)
    num_cores = mtp.cpu_count()
    print(f"num_cores: {num_cores}")
    # Use slightly fewer cores than available to avoid overloading the system
    num_segments = max(2, num_cores - 1)
    detector = SignDetector()

    # Try to load existing model first
    if not detector.load_model():
        print("No existing model found or error loading model")

        # Select and process a new reference image
        if detector.select_reference_image():
            print(f"Selected reference image: {detector.reference_image_path}")

            if detector.process_reference_image():
                detector.save_model()
            else:
                print("Failed to process reference image")
                return
        else:
            print("No reference image selected, exiting")
            return

    # Process video in parallel
    timestamps_start, timestamps_end = process_video_parallel(video_path, num_segments, start_sign, end_sign)

    # Process the results as before
    print("starts after normalize: ")
    print(timestamps_start)
    print("end: ")
    print(timestamps_end)

    app.date_time.set(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if not timestamps_start:
        print("No start points found")
        app.show_results()
        return

    # Convert to string with a specific format
    datetime_string = app.date_time.get()

    # Create clips from the timestamps
    folder_path = f"videos/{datetime_string}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    subclip_tasks = []
    segment_name = 1
    for idx, point_to_start in enumerate(timestamps_start):
        if len(timestamps_start) >= 2 and len(timestamps_start) > idx + 1:
            subclip_tasks.append(
                (video_path, point_to_start, timestamps_start[idx + 1], f"{folder_path}/{segment_name}.mp4"))
            segment_name = segment_name + 1

    print(f"start point of last clip: {timestamps_start[-1]}")
    start_time_last = timestamps_start[-1]

    if len(timestamps_end) > 0:
        print(f"end point of last clip: {timestamps_end[0]}")
        end_time_last = timestamps_end[0]
        subclip_tasks.append(
            (video_path, start_time_last, end_time_last, f"{folder_path}/{segment_name}.mp4"))
    else:
        subclip_tasks.append(
            (video_path, start_time_last, VideoFileClip(video_path).duration, f"{folder_path}/{segment_name}.mp4"))

    with mtp.Pool() as pool:
        pool.map(subclip, subclip_tasks)

    print(f"All {len(subclip_tasks)} clips processed in parallel")

    app.show_results()
    # APP ENDS #################################################

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":

    # run_with_camera()
    # Set appearance mode and default color theme for CustomTkinter
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    # Main application window
    root = TkinterDnD.Tk()
    root.title("ELS - E-commerce Livestream Segmentor")
    root.geometry("800x700")
    root.resizable(True, True)

    app = VideoSplitterApp(root)
    root.mainloop()
