import cv2
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

ref_signs_folder = f"./sign_detector/signs"
sign_filename = "sign.png"
model_folder = f"./sign_detector/model"
model_filename = "sign_detector_model.pkl"
ref_keypoints_folder = f"./sign_detector/keypoints"
ref_keypoints_filename = "reference_keypoints.png"
grid_vis_filename = "grid_visualization.png"


class SignDetector:
    def __init__(self):
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(ref_keypoints_folder):
            os.makedirs(ref_keypoints_folder)
        if not os.path.exists(ref_signs_folder):
            os.makedirs(ref_signs_folder)
        self.model_path = f"{model_folder}/{model_filename}"
        self.reference_image_path = os.path.join(ref_signs_folder, sign_filename)
        self.reference_image = None

        # Use a combination of detectors optimized for speed
        self.detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2,
                                       WTA_K=2, edgeThreshold=31)

        # Use FLANN for faster matching with large datasets
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # BFMatcher as fallback for when FLANN fails
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.min_match_count = 8
        self.ratio_threshold = 0.75  # For Lowe's ratio test
        self.match_threshold = 0.6
        self.loaded_model = False

        # Store keypoints and descriptors
        self.reference_keypoints = None
        self.reference_descriptors = None

        # Grid-based feature extraction parameters
        self.grid_size = (6, 6)  # 6x6 grid (reduced from 8x8 for speed)
        self.cell_keypoints = []
        self.cell_descriptors = []

        # Color histogram parameters (simplified for speed)
        self.histogram_bins = 16  # Reduced from 32 for speed
        self.reference_histograms = []
        self.hist_grid_size = (2, 2)  # Reduced from 4x4 for speed

        # Pre-compute scales for multi-scale matching
        self.scales = [0.75, 1.0, 1.25]  # Reduced number of scales for speed
    def select_reference_image(self):
        """Open a file dialog to select reference sign image"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(
            title="Select Sign Reference Image (with background already removed)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        root.destroy()

        if file_path:
            self.reference_image_path = file_path
            return True
        return False

    def process_reference_image(self):
        """Process reference image and extract features"""
        if not self.reference_image_path:
            print("No reference image selected")
            return False

        self.reference_image = cv2.imread(self.reference_image_path, cv2.IMREAD_COLOR)
        if self.reference_image is None:
            print(f"Failed to load reference image: {self.reference_image_path}")
            return False

        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 1. Extract global ORB features
        self.reference_keypoints, self.reference_descriptors = self.detector.detectAndCompute(enhanced, None)

        # 2. Extract grid-based features
        self._extract_grid_features(enhanced)

        # 3. Extract color histograms (for color robustness)
        self._extract_color_histograms(self.reference_image)

        # Save reference image
        cv2.imwrite(f"{ref_signs_folder}/{sign_filename}", self.reference_image)

        # Create visualization of keypoints (optional)
        keypoints_image = cv2.drawKeypoints(self.reference_image, self.reference_keypoints, None,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f"{ref_keypoints_folder}/{ref_keypoints_filename}", keypoints_image)

        print(f"Processed reference image with:")
        print(f"- {len(self.reference_keypoints)} ORB features")
        print(f"- {len(self.cell_descriptors)} grid cells")
        print(f"- {len(self.reference_histograms)} color histograms")

        return True

    def _extract_grid_features(self, gray_image):
        """Extract features from a grid covering the entire image"""
        h, w = gray_image.shape
        cell_h, cell_w = h // self.grid_size[0], w // self.grid_size[1]

        self.cell_descriptors = []
        self.cell_keypoints = []

        # For each cell in the grid
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Extract cell region
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < self.grid_size[0] - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < self.grid_size[1] - 1 else w

                cell = gray_image[y_start:y_end, x_start:x_end]

                # Detect keypoints in this cell
                cell_kps = cv2.goodFeaturesToTrack(cell, maxCorners=15, qualityLevel=0.01,
                                                   minDistance=5, blockSize=7)

                if cell_kps is not None:
                    cell_kps = [cv2.KeyPoint(x=pt[0][0] + x_start, y=pt[0][1] + y_start, size=5)
                                for pt in cell_kps]

                    # Extract descriptors for these keypoints
                    if len(cell_kps) > 0:
                        _, descs = self.detector.compute(gray_image, cell_kps)
                        if descs is not None:
                            self.cell_keypoints.append(cell_kps)
                            self.cell_descriptors.append(descs)

    def _extract_color_histograms(self, image):
        """Extract color histograms from grid cells for color matching"""
        h, w = image.shape[:2]
        cell_h, cell_w = h // self.hist_grid_size[0], w // self.hist_grid_size[1]

        self.reference_histograms = []

        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # For each cell in the grid
        for i in range(self.hist_grid_size[0]):
            for j in range(self.hist_grid_size[1]):
                # Extract cell region
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < self.hist_grid_size[0] - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < self.hist_grid_size[1] - 1 else w

                cell = hsv[y_start:y_end, x_start:x_end]

                # Calculate histogram for this cell
                hist = cv2.calcHist([cell], [0, 1], None,
                                    [self.histogram_bins, self.histogram_bins],
                                    [0, 180, 0, 256])

                # Normalize histogram
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                self.reference_histograms.append(hist)

    def save_model(self):
        """Save the reference features to a file"""
        if self.reference_keypoints is None or self.reference_descriptors is None:
            print("No features to save")
            return False

        # Convert keypoints to serializable format
        keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                                  for kp in self.reference_keypoints]

        # Convert cell keypoints
        cell_keypoints_serializable = []
        for cell_kps in self.cell_keypoints:
            cell_keypoints_serializable.append([
                (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in cell_kps
            ])

        model_data = {
            'image_path': self.reference_image_path,
            'keypoints': keypoints_serializable,
            'descriptors': self.reference_descriptors,
            'cell_keypoints': cell_keypoints_serializable,
            'cell_descriptors': self.cell_descriptors,
            'reference_histograms': self.reference_histograms,
            'grid_size': self.grid_size,
            'hist_grid_size': self.hist_grid_size,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {self.model_path}")
        return True

    def load_model(self):
        """Load the reference features from a file"""
        if not os.path.exists(self.model_path):
            print(f"Model file {self.model_path} does not exist")
            return False

        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.reference_image_path = model_data['image_path']
            self.grid_size = model_data['grid_size']
            self.hist_grid_size = model_data['hist_grid_size']

            # Convert serialized keypoints back to cv2.KeyPoint objects
            self.reference_keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1],
                                                     size=pt[1], angle=pt[2],
                                                     response=pt[3], octave=pt[4],
                                                     class_id=pt[5])
                                        for pt in model_data['keypoints']]

            self.reference_descriptors = model_data['descriptors']

            # Load cell keypoints and descriptors
            self.cell_keypoints = []
            for cell_kps in model_data['cell_keypoints']:
                self.cell_keypoints.append([
                    cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=pt[1], angle=pt[2],
                                 response=pt[3], octave=pt[4], class_id=pt[5])
                    for pt in cell_kps
                ])

            self.cell_descriptors = model_data['cell_descriptors']

            # Load color histograms
            self.reference_histograms = model_data['reference_histograms']

            # Load the reference image for display purposes
            if os.path.exists(self.reference_image_path):
                self.reference_image = cv2.imread(self.reference_image_path, cv2.IMREAD_COLOR)
            else:
                print(f"Reference image {self.reference_image_path} not found, but model loaded")

            print(f"Model loaded from {self.model_path} (created on {model_data['timestamp']})")
            print(f"Loaded {len(self.reference_keypoints)} features")

            self.loaded_model = True
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def detect_sign(self, frame):
        """
        Detect the sign in a frame using optimized feature matching

        Args:
            frame: Input frame (RGB or BGR)

        Returns:
            tuple: (confidence, matches, frame_keypoints)
        """
        if self.reference_descriptors is None or len(self.reference_keypoints) == 0:
            return 0.0, [], []

        # Ensure frame is in BGR format (for OpenCV)
        if frame.shape[2] == 3 and np.max(frame) > 1.0:
            # Assuming RGB format from your processing function
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame.copy()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Track best results across scales
        best_confidence = 0.0
        best_matches = []
        best_frame_keypoints = []

        # Try different scales (fewer scales for performance)
        for scale in self.scales:
            # Resize the frame according to scale
            h, w = enhanced.shape
            if scale != 1.0:
                scaled_img = cv2.resize(enhanced, (int(w * scale), int(h * scale)))
                scaled_color = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
            else:
                scaled_img = enhanced
                scaled_color = frame_bgr

            # Extract features from current scale
            try:
                # Regular ORB features
                frame_keypoints, frame_descriptors = self.detector.detectAndCompute(scaled_img, None)

                if frame_descriptors is None or len(frame_keypoints) < self.min_match_count:
                    continue

                # Try FLANN matcher first (faster)
                try:
                    matches = self.matcher.knnMatch(self.reference_descriptors, frame_descriptors, k=2)

                    # Apply Lowe's ratio test
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) < 2:
                            continue
                        m, n = match_pair
                        if m.distance < self.ratio_threshold * n.distance:
                            good_matches.append(m)
                except Exception as e:
                    # Fallback to BFMatcher if FLANN fails
                    matches = self.bf_matcher.knnMatch(self.reference_descriptors, frame_descriptors, k=2)

                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) < 2:
                            continue
                        m, n = match_pair
                        if m.distance < self.ratio_threshold * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= self.min_match_count:
                    # Try to find homography
                    src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1,
                                                                                                                  2)
                    dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if homography is not None:
                        # Calculate confidence based on inliers and match quality
                        inliers = mask.ravel().sum()
                        inlier_ratio = inliers / len(good_matches) if good_matches else 0

                        # Calculate average distance
                        avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                        max_distance = 100  # Typical max distance for ORB
                        distance_confidence = max(0, 1 - (avg_distance / max_distance))

                        # Add color similarity for more robust detection
                        color_confidence = self._color_similarity(scaled_color)

                        # Combine confidence factors
                        # Weigh inlier_ratio more heavily as it's the most reliable indicator
                        combined_confidence = (0.6 * inlier_ratio +
                                               0.3 * distance_confidence +
                                               0.1 * color_confidence)

                        # Track best results
                        if combined_confidence > best_confidence:
                            best_confidence = combined_confidence

                            # Scale keypoints back to original frame coordinates if needed
                            if scale != 1.0:
                                scaled_keypoints = []
                                for kp in frame_keypoints:
                                    new_kp = cv2.KeyPoint(x=kp.pt[0] / scale, y=kp.pt[1] / scale,
                                                          size=kp.size / scale, angle=kp.angle,
                                                          response=kp.response, octave=kp.octave,
                                                          class_id=kp.class_id)
                                    scaled_keypoints.append(new_kp)
                                best_frame_keypoints = scaled_keypoints
                            else:
                                best_frame_keypoints = frame_keypoints

                            best_matches = good_matches
            except Exception as e:
                # Print error but continue with next scale
                print(f"Error in scale {scale}: {str(e)}")
                continue

        return best_confidence, best_matches, best_frame_keypoints

    def _color_similarity(self, frame):
        """Calculate color similarity between reference and frame"""
        if not self.reference_histograms:
            return 0.0

        h, w = frame.shape[:2]
        cell_h, cell_w = h // self.hist_grid_size[0], w // self.hist_grid_size[1]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        similarities = []

        # For each cell
        for i in range(self.hist_grid_size[0]):
            for j in range(self.hist_grid_size[1]):
                # Get cell coordinates
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < self.hist_grid_size[0] - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < self.hist_grid_size[1] - 1 else w

                # Extract region
                cell = hsv[y_start:y_end, x_start:x_end]

                # Calculate histogram
                hist = cv2.calcHist([cell], [0, 1], None,
                                    [self.histogram_bins, self.histogram_bins],
                                    [0, 180, 0, 256])

                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                # Get index in reference histograms
                idx = i * self.hist_grid_size[1] + j
                if idx < len(self.reference_histograms):
                    # Compare histograms
                    similarity = cv2.compareHist(self.reference_histograms[idx], hist, cv2.HISTCMP_CORREL)
                    similarities.append(similarity)

        # Average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    def run_camera_detection(self):
        """Run sign detection using the webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\n===== CAMERA DETECTION STARTED =====")
        print("Press 'q' to quit")
        print("Press 's' to save a screenshot")
        print("Press 't'/'T' to decrease/increase detection threshold")
        print(f"Current threshold: {self.match_threshold:.2f}")
        print("===================================\n")

        # Keep track of recent confidence scores for smoothing
        confidence_history = []
        history_size = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            display_frame = frame.copy()

            # Detect sign with best matches and keypoints
            confidence, matches, frame_keypoints = self.detect_sign(frame)

            # Add to history and maintain fixed size
            confidence_history.append(confidence)
            if len(confidence_history) > history_size:
                confidence_history.pop(0)

            # Use average confidence for stability
            avg_confidence = sum(confidence_history) / len(confidence_history)

            # Display confidence value in corner for debugging
            cv2.putText(display_frame, f"Conf: {avg_confidence:.2f}",
                        (display_frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # If sign is detected with good confidence
            if avg_confidence >= self.match_threshold:
                # Draw green border around entire frame
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 10)

                # Add text to the frame
                cv2.putText(display_frame, "Sign Detected!",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw matches if available
                if len(matches) > 0 and self.reference_image is not None and len(frame_keypoints) > 0:
                    try:
                        # Draw top 15 matches in a separate window for visualization
                        top_matches = sorted(matches, key=lambda x: x.distance)[:15]
                        matches_img = cv2.drawMatches(
                            self.reference_image, self.reference_keypoints,
                            frame, frame_keypoints,
                            top_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                        cv2.imshow('Sign Matches', matches_img)
                    except Exception as e:
                        print(f"Error drawing matches: {str(e)}")

            # Add threshold info at bottom
            cv2.putText(display_frame, f"Threshold: {self.match_threshold:.2f} (t/T to adjust)",
                        (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the frame
            cv2.imshow('Sign Detector', display_frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"sign_detection_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"Screenshot saved to {screenshot_path}")
            elif key == ord('t'):
                self.match_threshold = max(0.1, self.match_threshold - 0.05)
                print(f"Reduced confidence threshold to: {self.match_threshold:.2f}")
            elif key == ord('T'):
                self.match_threshold = min(0.9, self.match_threshold + 0.05)
                print(f"Increased confidence threshold to: {self.match_threshold:.2f}")

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    def initialize(self):
        # Try to load existing model first
        if not self.load_model():
            if self.process_reference_image():
                self.save_model()
                return True
            else:
                messagebox.showwarning("No Sign Found", "No existing sign or error loading sign!")
                return False
        return True


def process_segment_with_sign(start_time, end_time, video_path, visualize=True):
    """
    Process a specific segment of the video, focusing only on frame-by-frame detection.
    Records every timestamp where a sign is detected.

    Args:
        video_path (str): Path to the video file
        start_time (float): Time in seconds to start processing
        end_time (float): Time in seconds to end processing
        visualize (bool): Whether to show frame-by-frame visualization

    Returns:
        tuple: Lists of timestamps for detected signs and empty end timestamps list
    """
    from ELS import calculate_frames_to_skip
    cap = cv2.VideoCapture(video_path)
    detector = SignDetector()

    # Make sure model is properly loaded
    if not detector.load_model():
        print("Error: Failed to load sign detection model")
        return [], []

    if not cap.isOpened():
        print(f"Error: Could not open video for segment {start_time}-{end_time}.")
        return [], []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    timestamps_start = []  # Will contain all timestamps where a sign is detected
    timestamps_end = []  # Will be returned empty as requested
    frame_skip = calculate_frames_to_skip(frame_rate)

    print(f"Processing segment from {start_time:.2f}s to {end_time:.2f}s")
    print(f"Frame skip: {frame_skip}")

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame or based on frame_skip
        if frame_count % frame_skip == 0:
            timestamp = frame_count / frame_rate

            # Create a display copy of the frame
            if visualize:
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                display_frame = cv2.resize(display_frame, (w // 2, h // 2))

            # Detect sign - direct raw confidence
            confidence, matches, frame_keypoints = detector.detect_sign(frame)

            # Debug information
            print(f"Frame {frame_count}, Time: {timestamp:.2f}s, Confidence: {confidence:.2f}")

            # Simple frame-by-frame detection
            if confidence >= detector.match_threshold:
                # Record this timestamp
                timestamps_start.append(timestamp)
                print(f"Sign detected at {timestamp:.2f}s with confidence {confidence:.2f}")

            # Visualization logic (reduced size)
            if visualize:
                # Display confidence value in corner
                cv2.putText(display_frame, f"Conf: {confidence:.2f}",
                            (display_frame.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # If sign is detected with good confidence
                if confidence >= detector.match_threshold:
                    # Draw green border around entire frame
                    h, w = display_frame.shape[:2]
                    cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 10)

                    # Add text to the frame
                    cv2.putText(display_frame, "Sign Detected!",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Draw matches if available
                    if len(matches) > 0 and detector.reference_image is not None and len(frame_keypoints) > 0:
                        try:
                            # Draw top matches in a separate window
                            top_matches = sorted(matches, key=lambda x: x.distance)[:15]
                            matches_img = cv2.drawMatches(
                                detector.reference_image, detector.reference_keypoints,
                                frame, frame_keypoints,
                                top_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                            )
                            matches_img = cv2.resize(matches_img,
                                                     (matches_img.shape[1] // 2, matches_img.shape[0] // 2))
                            cv2.imshow('Sign Matches', matches_img)
                        except Exception as e:
                            print(f"Error drawing matches: {str(e)}")

                # Add threshold info at bottom
                cv2.putText(display_frame, f"Threshold: {detector.match_threshold:.2f}",
                            (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show the frame
                cv2.imshow('Processing Segment', display_frame)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord('t'):
                    detector.match_threshold = max(0.1, detector.match_threshold - 0.05)
                    print(f"Reduced threshold to: {detector.match_threshold:.2f}")
                elif key == ord('T'):
                    detector.match_threshold = min(0.9, detector.match_threshold + 0.05)
                    print(f"Increased threshold to: {detector.match_threshold:.2f}")

        frame_count += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    print(f"Found {len(timestamps_start)} frames with sign detected")
    return timestamps_start, timestamps_end