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

        # Use a combination of ORB and SIFT-like detectors for better scale invariance
        # ORB is fast but AKAZE offers better scale invariance
        self.detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2,
                                       WTA_K=2, edgeThreshold=31)

        # Use ratio test matching for better results with scale changes
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.min_match_count = 5
        self.ratio_threshold = 0.75  # For Lowe's ratio test
        self.match_threshold = 0.35
        self.loaded_model = False

        # Store keypoints and descriptors
        self.reference_keypoints = None
        self.reference_descriptors = None

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

        # Create an image pyramid for multi-scale feature extraction
        scales = [1.0, 0.75, 0.5]  # Multiple scales to improve detection at different distances
        all_keypoints = []
        all_descriptors = []

        for scale in scales:
            # Resize image according to scale
            if scale != 1.0:
                h, w = enhanced.shape
                scaled_img = cv2.resize(enhanced, (int(w * scale), int(h * scale)))
            else:
                scaled_img = enhanced

            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.detector.detectAndCompute(scaled_img, None)

            if descriptors is not None and len(keypoints) > 0:
                # Adjust keypoint coordinates back to original image scale
                if scale != 1.0:
                    for kp in keypoints:
                        kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

                all_keypoints.extend(keypoints)
                if len(all_descriptors) == 0:
                    all_descriptors = descriptors
                else:
                    all_descriptors = np.vstack((all_descriptors, descriptors))

        # Filter out redundant keypoints
        if len(all_keypoints) > 0:
            self.reference_keypoints = all_keypoints
            self.reference_descriptors = all_descriptors
        else:
            print("No features detected in reference image. Try a different image with more distinctive features.")
            return False

        print(f"Extracted {len(self.reference_keypoints)} features from reference image")

        cv2.imwrite(f"{ref_signs_folder}/{sign_filename}", self.reference_image)
        # Draw keypoints on reference image and show
        keypoints_image = cv2.drawKeypoints(self.reference_image, self.reference_keypoints, None,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f"{ref_keypoints_folder}/{ref_keypoints_filename}", keypoints_image)

        # Show images for visual inspection
        # cv2.imshow("Reference Image", self.reference_image)
        # cv2.imshow("Reference Keypoints", keypoints_image)
        # print("Press any key to continue...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return True

    def save_model(self):
        """Save the reference features to a file"""
        if self.reference_keypoints is None or self.reference_descriptors is None:
            print("No features to save")
            return False

        # Convert keypoints to serializable format
        keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                                  for kp in self.reference_keypoints]

        model_data = {
            'image_path': self.reference_image_path,
            'keypoints': keypoints_serializable,
            'descriptors': self.reference_descriptors,
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

            # Convert serialized keypoints back to cv2.KeyPoint objects
            self.reference_keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1],
                                                     size=pt[1], angle=pt[2],
                                                     response=pt[3], octave=pt[4],
                                                     class_id=pt[5])
                                        for pt in model_data['keypoints']]

            self.reference_descriptors = model_data['descriptors']

            # Load the reference image for display purposes
            if os.path.exists(self.reference_image_path):
                self.reference_image = cv2.imread(self.reference_image_path, cv2.IMREAD_COLOR)
            else:
                print(f"Reference image {self.reference_image_path} not found, but model loaded")

            print(f"Model loaded from {self.model_path} (created on {model_data['timestamp']})")
            print(f"Loaded {len(self.reference_keypoints)} features")

            if self.reference_image is None:
                print("Reference Image is not available")

            self.loaded_model = True
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def detect_sign(self, frame):
        """Detect the sign in a frame using feature matching with scale invariance"""
        if self.reference_descriptors is None or len(self.reference_keypoints) == 0:
            return 0.0, []

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Create a multi-scale pyramid for better detection at various distances
        scales = [1.0, 0.75, 0.5, 1.5]  # Added 1.5 for when sign is closer
        best_confidence = 0.0
        best_matches = []
        best_frame_keypoints = []

        for scale in scales:
            # Resize the frame according to scale
            h, w = enhanced.shape
            if scale != 1.0:
                scaled_img = cv2.resize(enhanced, (int(w * scale), int(h * scale)))
            else:
                scaled_img = enhanced

            # Detect features in the scaled frame
            frame_keypoints, frame_descriptors = self.detector.detectAndCompute(scaled_img, None)

            if frame_descriptors is None or len(frame_keypoints) < self.min_match_count:
                continue

            # Apply ratio test matching for better matching
            try:
                # Find the 2 best matches for each descriptor
                matches = self.matcher.knnMatch(self.reference_descriptors, frame_descriptors, k=2)

                # Apply Lowe's ratio test to filter good matches
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) < 2:
                        continue
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)

                if len(good_matches) >= self.min_match_count:
                    # Calculate confidence based on number and quality of matches
                    avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                    max_distance = 100  # Typical max distance for ORB

                    # Convert to confidence score (0 to 1)
                    confidence = max(0, 1 - (avg_distance / max_distance))

                    # Weight confidence by number of matches found
                    match_factor = min(1.0, len(good_matches) / 30)  # Cap at 30 matches
                    scaled_confidence = confidence * match_factor

                    # Track best result across all scales
                    if scaled_confidence > best_confidence:
                        best_confidence = scaled_confidence
                        best_matches = good_matches

                        # Convert keypoint coordinates back to original image scale
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

            except Exception as e:
                print(f"Error during matching at scale {scale}: {str(e)}")

        return best_confidence, best_matches, best_frame_keypoints

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

def process_segment_with_sign(start_time, end_time, video_path, start_sign, end_sign):
    """
    Process a specific segment of the video.

    Args:
        video_path (str): Path to the video file
        start_time (float): Time in seconds to start processing
        end_time (float): Time in seconds to end processing

    Returns:
        tuple: Lists of timestamps for starts and ends
    """
    from ELS import time_between_batches, pose_duration, initialize_dynamic_parameters, calculate_frames_to_skip
    cap = cv2.VideoCapture(video_path)

    detector = SignDetector()

    # Try to load existing model first
    detector.load_model()

    if not cap.isOpened():
        print(f"Error: Could not open video for segment {start_time}-{end_time}.")
        return [], []

    # Get video details
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Set starting position
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    timestamps_start = []
    timestamps_end = []

    # initialize_dynamic_parameters(frame_rate, 1.5)
    frame_skip = calculate_frames_to_skip(frame_rate)
    print(f"Frame skip: {frame_skip}")
    end_detected = False
    # Process frames until we reach the end frame
    while frame_count < end_frame and not end_detected:
        ret, image = cap.read()
        if not ret:
            break

        # Skip frames for speed
        if frame_count % frame_skip == 0:

            timestamp = frame_count / frame_rate
            # print(f"{timestamp}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Keep track of recent confidence scores for smoothing
            confidence_history = []
            history_size = 5

            # Detect sign with best matches and keypoints
            confidence, matches, frame_keypoints = detector.detect_sign(image)

            # Add to history and maintain fixed size
            confidence_history.append(confidence)
            if len(confidence_history) > history_size:
                confidence_history.pop(0)

            # Use average confidence for stability
            avg_confidence = sum(confidence_history) / len(confidence_history)

            # If sign is detected with good confidence
            if avg_confidence >= detector.match_threshold:
                timestamps_start.append(timestamp)
        frame_count += 1
    cap.release()
    return timestamps_start, timestamps_end

