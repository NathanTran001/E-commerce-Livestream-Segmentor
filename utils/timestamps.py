import cv2 as cv


def normalize_timestamps(timestamps, time_between_batches, pose_duration):
    """
    Process timestamps and return normalized timestamps that represent valid pose completions.

    A valid pose is determined when a batch of timestamps spans at least 'pose_duration'.
    Timestamps are grouped into batches where each adjacent pair has a gap less than 'time_between_batches'.
    For each valid batch, the last timestamp is returned as the normalized timestamp.

    Args:
        timestamps: List of timestamp floats to process
        time_between_batches: Minimum time gap that separates batches
        pose_duration: Minimum duration a batch must span to be considered valid

    Returns:
        List of normalized timestamps (one per valid batch)
    """
    if not timestamps or len(timestamps) < 2:
        return []

    normalized_timestamps = []
    timestamps.sort()
    print(f"TS A: {timestamps}")

    current_batch = [timestamps[0]]  # Start with the first timestamp

    for i in range(1, len(timestamps)):
        current_timestamp = timestamps[i]
        previous_timestamp = timestamps[i - 1]

        # Check if this timestamp belongs to the current batch
        if abs(current_timestamp - previous_timestamp) < time_between_batches:
            # Same batch, add to current batch
            current_batch.append(current_timestamp)
        else:
            # Different batch, process the previous batch if valid
            if len(current_batch) > 1 and (current_batch[-1] - current_batch[0]) >= pose_duration:
                normalized_timestamps.append(current_batch[-1])

            # Start a new batch with the current timestamp
            current_batch = [current_timestamp]

    # Process the final batch
    if len(current_batch) > 1 and (current_batch[-1] - current_batch[0]) >= pose_duration:
        normalized_timestamps.append(current_batch[-1])

    print(f"TS B: {normalized_timestamps}")
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

