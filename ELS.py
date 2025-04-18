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
from utils.hand_gesture import calc_landmark_list, pre_process_landmark, run_with_camera, process_video_parallel
from utils.timestamps import normalize_timestamps, calculate_segment_boundaries

pose_duration = 1
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
    subclip_tasks = []
    segment_name = 1
    for idx, point_to_start in enumerate(timestamps_start):
        if len(timestamps_start) >= 2 and len(timestamps_start) > idx + 1:
            subclip_tasks.append(
                (datetime_string, video_path, point_to_start, timestamps_start[idx + 1], f"{segment_name}.mp4"))
            segment_name = segment_name + 1

    print(f"start point of last clip: {timestamps_start[-1]}")
    start_time_last = timestamps_start[-1]

    if len(timestamps_end) > 0:
        print(f"end point of last clip: {timestamps_end[0]}")
        end_time_last = timestamps_end[0]
        subclip_tasks.append(
            (datetime_string, video_path, start_time_last, end_time_last, f"{segment_name}.mp4"))
    else:
        subclip_tasks.append(
            (datetime_string, video_path, start_time_last, VideoFileClip(video_path).duration, f"{segment_name}.mp4"))

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
