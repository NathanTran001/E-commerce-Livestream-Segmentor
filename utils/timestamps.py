import cv2 as cv


def normalize_timestamps(timestamps, time_between_batches, pose_duration):
    temp_timestamps = []
    normalized_timestamps = []
    for idx, timestamp in enumerate(timestamps):  # 2.1 2.2 2.3   5.6 5.7   8.1 8.2 8.3 8.4 8.5
        # When haven't reached the end
        if len(timestamps) > idx + 1 and len(timestamps) >= 2:
            timestamps.sort()
            # Keep flushing to temp
            temp_timestamps.append(timestamp)
            # When reach a different batch
            if abs(timestamp - timestamps[idx + 1]) > time_between_batches:
                # Validate current temp batch and clean temp
                if (len(temp_timestamps) > 0 and
                        temp_timestamps[-1] - temp_timestamps[0] >= pose_duration):
                    normalized_timestamps.append(temp_timestamps[-1])
                temp_timestamps.clear()
        # When reach the end
        elif idx + 1 == len(timestamps) >= 2:
            # Only process if it is a same-batch stamp since a different-batch stamp being the final stamp is discarded
            # If same batch -> Still add that final stamp to temp
            if (len(temp_timestamps) > 0 and
                    abs(timestamp - temp_timestamps[-1]) <= time_between_batches):
                temp_timestamps.append(timestamp)
            # Validate current batch and clean temp
            if (len(temp_timestamps) > 0 and
                    temp_timestamps[-1] - temp_timestamps[0] >= pose_duration):
                normalized_timestamps.append(temp_timestamps[-1])
            temp_timestamps.clear()
    return normalized_timestamps


def calculate_segment_boundaries(video_path, num_segments):
    """
    Calculate the starting and ending points for each segment of the video.

    Args:
        video_path (str): Path to the video file
        num_segments (int): Number of segments to divide the video into

    Returns:
        list of tuples: Each tuple contains (start_time, end_time) in seconds
    """
    # Open the video to get its duration
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # Get total frames and frame rate
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    total_duration = total_frames / frame_rate

    cap.release()

    # Calculate segment duration
    segment_duration = total_duration / num_segments

    # Calculate segment boundaries
    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        # For the last segment, ensure we go to the exact end
        if i == num_segments - 1:
            end_time = total_duration

        segments.append((start_time, end_time))

    return segments

