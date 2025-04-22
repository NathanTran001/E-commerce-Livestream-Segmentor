import os
import sys
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from tkinterdnd2 import *
from PIL import Image, ImageTk
import cv2

from utils import sign_detector
from utils.sign_detector import ref_signs_folder, sign_filename, ref_keypoints_folder, ref_keypoints_filename, \
    model_folder, model_filename, SignDetector

from utils.colors import DEFAULT

start_sign = "Select Start Gesture"
end_sign = "Select End Gesture"
sign_path = os.path.join(ref_signs_folder, sign_filename)
model_path = os.path.join(model_folder, model_filename)
keypoint_path = os.path.join(ref_keypoints_folder, ref_keypoints_filename)


class VideoSplitterApp:
    """Application for splitting videos into multiple shorter clips.

    This class provides a CustomTkinter-based GUI for uploading videos
    and splitting them into shorter segments using either hand gestures
    or custom sign images.
    """

    def __init__(self, master):
        """Initialize the application.

        Args:
            master: The root CTk window of the application.
        """
        self.master = master
        self.selected_file = tk.StringVar()
        self.date_time = tk.StringVar()
        self.start = tk.StringVar(value="Select Start Gesture")
        self.end = tk.StringVar(value="Select End Gesture")
        self.mode = tk.StringVar(value="hand_sign")  # Default mode
        self.custom_start_image = None  # Path to custom start sign image
        self.custom_end_image = None  # Path to custom end sign image
        self.sign_detector = SignDetector()
        self.gesture_icons = {}
        self.gesture_pil_images = {}
        self.output_folder = tk.StringVar()

        # Set up frames for different screens
        self._setup_frames()
        self.check_for_saved_signs()

        # Start with the home frame visible
        self.show_frame(self.home_frame)

        # Add trace to update globals when StringVars change
        self.start.trace_add("write", self._update_global_start)
        self.end.trace_add("write", self._update_global_end)

    def _update_global_start(self, *args):
        global start_sign
        start_sign = self.start.get()
        print(f"Global start_sign updated to: {start_sign}")

    def _update_global_end(self, *args):
        global end_sign
        end_sign = self.end.get()
        print(f"Global end_sign updated to: {end_sign}")

    def _setup_frames(self):
        """Set up the main frames for different application states."""
        # Create frames for different states with the same grid position
        self.home_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)
        self.hand_main_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)
        self.sign_main_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)
        self.loading_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)
        self.results_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)

        # Stack frames on top of each other
        for frame in (self.home_frame, self.hand_main_frame, self.sign_main_frame,
                      self.loading_frame, self.results_frame):
            frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid to allow expansion
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Set up each frame's content
        self._setup_home_frame()
        self._setup_hand_main_frame()
        self._setup_sign_main_frame()
        self._setup_loading_frame()
        self._setup_results_frame()

    def _setup_home_frame(self):
        """Set up the home frame with mode selection buttons."""
        # Configure frame layout
        self.home_frame.grid_columnconfigure(0, weight=1)
        self.home_frame.grid_rowconfigure(0, weight=1)  # Top space
        self.home_frame.grid_rowconfigure(1, weight=0)  # Title
        self.home_frame.grid_rowconfigure(2, weight=0)  # Description
        self.home_frame.grid_rowconfigure(3, weight=0)  # Hand sign button
        self.home_frame.grid_rowconfigure(4, weight=0)  # Custom sign button
        self.home_frame.grid_rowconfigure(5, weight=1)  # Bottom space

        # Title
        title_label = ctk.CTkLabel(
            self.home_frame,
            text="ELS",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#1A73E8"
        )
        title_label.grid(row=1, column=0, padx=20, pady=(20, 5))

        # Description
        desc_label = ctk.CTkLabel(
            self.home_frame,
            text="Choose a mode to split your videos",
            font=ctk.CTkFont(size=16),
            text_color="#606770"
        )
        desc_label.grid(row=2, column=0, padx=20, pady=(0, 20))

        # Mode buttons container
        buttons_frame = ctk.CTkFrame(self.home_frame, fg_color="#E8ECEF", corner_radius=0)
        buttons_frame.grid(row=3, column=0, rowspan=2, padx=20, pady=20)

        # Hand Sign Mode Button
        hand_sign_btn = ctk.CTkButton(
            buttons_frame,
            text="Hand Sign Mode",
            command=lambda: self._enter_hand_sign_mode(),
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        hand_sign_btn.grid(row=0, column=0, padx=20, pady=10)

        # Custom Sign Mode Button
        custom_sign_btn = ctk.CTkButton(
            buttons_frame,
            text="Custom Sign Mode",
            command=lambda: self._enter_custom_sign_mode(),
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        custom_sign_btn.grid(row=1, column=0, padx=20, pady=10)

        # Version info at bottom
        version_label = ctk.CTkLabel(
            self.home_frame,
            text="E-commerce Livestream Segmentor v2.0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        version_label.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="s")

    def _enter_custom_sign_mode(self):
        self.mode.set("custom_sign")
        self.show_frame(self.sign_main_frame)

    def _enter_hand_sign_mode(self):
        self.mode.set("hand_sign")
        self.show_frame(self.hand_main_frame)

    def _setup_hand_main_frame(self):
        # Configure frame layout
        self.hand_main_frame.grid_columnconfigure(0, weight=1)
        self.hand_main_frame.grid_rowconfigure(0, weight=0)  # Header
        self.hand_main_frame.grid_rowconfigure(1, weight=1)  # Drag area
        self.hand_main_frame.grid_rowconfigure(2, weight=0)  # File label
        self.hand_main_frame.grid_rowconfigure(4, weight=0)  # Execute button
        self.hand_main_frame.grid_rowconfigure(5, weight=0)  # Footer

        self.setup_main_frame_header(self.hand_main_frame, "Hand Sign Mode")

        # Create a new frame specifically for the split row
        upload_row_frame = tk.Frame(self.hand_main_frame)
        upload_row_frame.grid(row=1, column=0, sticky="nsew")

        # Configure the columns in the split row frame with 3:1 ratio
        upload_row_frame.grid_columnconfigure(0, weight=3)  # Left section (3 parts)
        upload_row_frame.grid_columnconfigure(1, weight=1)  # Right section (1 part)
        upload_row_frame.grid_rowconfigure(0, weight=1)  # Make the row expand

        self.setup_upload_row(upload_row_frame)
        self.setup_gesture_selection(upload_row_frame)

        # Selected file label
        file_label = ctk.CTkLabel(
            self.hand_main_frame,
            textvariable=self.selected_file,
            wraplength=700
        )
        file_label.grid(row=2, column=0, padx=20, pady=5)

        # Execute button (centered)
        execute_button = ctk.CTkButton(
            self.hand_main_frame,
            text="Execute",
            command=self.execute_split,
            width=150,
            font=ctk.CTkFont(weight="bold")
        )
        execute_button.grid(row=4, column=0, padx=20, pady=10)

        # Footer/info section (centered)
        footer_frame = ctk.CTkFrame(self.hand_main_frame, fg_color="#E8ECEF", corner_radius=0)
        footer_frame.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="ew")

        # Configure footer for centering
        footer_frame.grid_columnconfigure(0, weight=1)

        # Version info
        info_label = ctk.CTkLabel(
            footer_frame,
            text="E-commerce Livestream Segmentor v2.0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        info_label.grid(row=0, column=0, padx=20, pady=(5, 0))

        # Description
        desc_label = ctk.CTkLabel(
            footer_frame,
            text="Easily split your videos into shorter clips using hand signs!",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc_label.grid(row=1, column=0, padx=20, pady=(0, 5))

    def setup_main_frame_header(self, main_frame, title):
        # Create a new frame specifically for the split row
        header_row_frame = tk.Frame(main_frame)
        header_row_frame.grid(row=0, column=0, sticky="nsew")

        # Configure the columns in the split row frame with ratio
        header_row_frame.grid_columnconfigure(0, weight=1)
        header_row_frame.grid_columnconfigure(1, weight=10)
        header_row_frame.grid_columnconfigure(2, weight=1)
        header_row_frame.grid_rowconfigure(0, weight=1)  # Make the row expand

        # Back button (top left)
        back_button = ctk.CTkButton(
            header_row_frame,
            text="Back",
            command=lambda: self.show_frame(self.home_frame),
            width=80,
            height=30,
            fg_color=DEFAULT,
            text_color="#1A73E8",
            hover_color="#D0D7DE",
            border_width=0,
        )
        back_button.grid(row=0, column=0, padx=(20, 5), pady=(20, 10), sticky="nsew")
        # Mode indicator
        dummy_button = ctk.CTkButton(
            header_row_frame,
            text="Back",
            width=80,
            height=30,
            fg_color=DEFAULT,
            text_color=DEFAULT,
            hover_color=DEFAULT,
            border_width=0,
        )
        dummy_button.grid(row=0, column=2, padx=(5, 20), pady=(20, 10), sticky="nsew")

        mode_label_container = ctk.CTkFrame(header_row_frame, fg_color=DEFAULT)
        mode_label_container.grid(row=0, column=1, sticky="nsew")

        # Configure the container to enable centering
        mode_label_container.grid_columnconfigure(0, weight=1)
        mode_label_container.grid_rowconfigure(0, weight=1)

        # Mode indicator
        mode_label = ctk.CTkLabel(
            mode_label_container,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="center",
            text_color="#1A73E8",
        )
        mode_label.grid(row=0, column=0, pady=(20, 10), sticky="")  # Empty sticky centers the widget

    def _setup_sign_main_frame(self):
        """Set up the custom sign input frame with image upload functionality."""
        # Configure frame layout
        self.sign_main_frame.grid_columnconfigure(0, weight=1)
        self.sign_main_frame.grid_rowconfigure(0, weight=0)  # Header with back button
        self.sign_main_frame.grid_rowconfigure(1, weight=1)  # Drag area
        self.sign_main_frame.grid_rowconfigure(2, weight=0)  # File label
        self.sign_main_frame.grid_rowconfigure(4, weight=0)  # Execute button
        self.sign_main_frame.grid_rowconfigure(5, weight=0)  # Footer

        self.setup_main_frame_header(self.sign_main_frame, "Custom Sign Mode")

        # Create a new frame specifically for the split row
        upload_row_frame = tk.Frame(self.sign_main_frame)
        upload_row_frame.grid(row=1, column=0, sticky="nsew")

        # Configure the columns in the split row frame with 3:1 ratio
        upload_row_frame.grid_columnconfigure(0, weight=3)  # Left section (3 parts)
        upload_row_frame.grid_columnconfigure(1, weight=1)  # Right section (1 part)
        upload_row_frame.grid_rowconfigure(0, weight=1)  # Make the row expand

        self.setup_upload_row(upload_row_frame)

        custom_sign_upload_container = ctk.CTkFrame(upload_row_frame, fg_color=DEFAULT)
        custom_sign_upload_container.grid(row=0, column=1, sticky="nsew")  # Make it fill all space
        custom_sign_upload_container.grid_rowconfigure(1, weight=1)  # Row with start_sign_container
        custom_sign_upload_container.grid_columnconfigure(0, weight=1)  # Allow column to expand

        # Start sign container
        start_sign_container = ctk.CTkFrame(custom_sign_upload_container, fg_color="white", corner_radius=8)
        start_sign_container.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")  # Make it fill allocated space
        start_sign_container.grid_rowconfigure(1, weight=1)  # Give weight to image display row
        start_sign_container.grid_columnconfigure(0, weight=1)  # Allow column to expand

        # Start sign label
        start_sign_label = ctk.CTkLabel(
            start_sign_container,
            text="Your Sign",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        start_sign_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")  # Allow horizontal expansion

        # Start sign image placeholder/display area
        self.start_sign_display = ctk.CTkFrame(
            start_sign_container,
            width=150,
            height=150,
            fg_color="#E8ECEF",
            corner_radius=8
        )
        self.start_sign_display.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="nsew")  # Fill all available space

        # This will store the label or image widget for the start sign
        self.start_sign_widget = ctk.CTkLabel(
            self.start_sign_display,
            text="No image selected",
            text_color="gray"
        )
        self.start_sign_widget.place(relx=0.5, rely=0.5, anchor="center")  # Keep this centered

        # Upload button for start sign
        start_upload_btn = ctk.CTkButton(
            start_sign_container,
            text="Upload Your Sign",
            command=lambda: self.select_custom_sign("start"),
            font=ctk.CTkFont(size=12)
        )
        start_upload_btn.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="")

        # Selected file label
        file_label = ctk.CTkLabel(
            self.sign_main_frame,
            textvariable=self.selected_file,
            wraplength=700
        )
        file_label.grid(row=2, column=0, padx=20, pady=5)

        # Execute button (centered)
        execute_button = ctk.CTkButton(
            self.sign_main_frame,
            text="Execute",
            command=self.execute_split,
            width=150,
            font=ctk.CTkFont(weight="bold")
        )
        execute_button.grid(row=4, column=0, padx=20, pady=10)

        # Footer/info section (centered)
        footer_frame = ctk.CTkFrame(self.sign_main_frame, fg_color="#E8ECEF", corner_radius=0)
        footer_frame.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="ew")

        # Configure footer for centering
        footer_frame.grid_columnconfigure(0, weight=1)

        # Version info
        info_label = ctk.CTkLabel(
            footer_frame,
            text="E-commerce Livestream Segmentor v2.0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        info_label.grid(row=0, column=0, padx=20, pady=(5, 0))

        # Description
        desc_label = ctk.CTkLabel(
            footer_frame,
            text="Easily split your videos using custom sign images!",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc_label.grid(row=1, column=0, padx=20, pady=(0, 5))

    def setup_upload_row(self, upload_row_frame):
        # Drag-and-drop area (same as input_frame)
        drag_area = ctk.CTkFrame(upload_row_frame, fg_color="white", corner_radius=8)
        drag_area.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="nsew")

        # Configure drag area to center content
        drag_area.grid_rowconfigure(0, weight=1)  # Space above content
        drag_area.grid_rowconfigure(1, weight=0)  # Content container
        drag_area.grid_rowconfigure(2, weight=1)  # Space below content
        drag_area.grid_columnconfigure(0, weight=1)

        # Create a container frame to hold all three elements together
        content_container = ctk.CTkFrame(drag_area, fg_color="transparent")
        content_container.grid(row=1, column=0, sticky="")

        # Configure the content container
        content_container.grid_columnconfigure(0, weight=1)
        content_container.grid_rowconfigure(0, weight=0)  # Icon
        content_container.grid_rowconfigure(1, weight=0)  # Main label
        content_container.grid_rowconfigure(2, weight=0)  # Sublabel

        # Upload icon
        upload_icon = ctk.CTkLabel(
            content_container,
            text="⬆",
            font=ctk.CTkFont(size=40),
            text_color="#A9A9A9"
        )
        upload_icon.grid(row=0, column=0, padx=20, pady=(0, 5))

        # Text labels
        drag_label = ctk.CTkLabel(
            content_container,
            text="Select video to upload",
            font=ctk.CTkFont(size=16)
        )
        drag_label.grid(row=1, column=0, padx=20, pady=(0, 2))

        drag_sublabel = ctk.CTkLabel(
            content_container,
            text="Or drag and drop video files",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        drag_sublabel.grid(row=2, column=0, padx=20, pady=(0, 0))

        # Enable drag-and-drop
        drag_area.drop_target_register(DND_FILES)
        drag_area.dnd_bind('<<Drop>>', self.handle_drop)

        # Make everything clickable to open file dialog
        for widget in [upload_icon, drag_label]:
            widget.bind("<Button-1>", lambda event: self.select_video())
            widget.bind("<Enter>", lambda event: widget.configure(cursor="hand2"))
            widget.bind("<Leave>", lambda event: widget.configure(cursor=""))

    def setup_gesture_selection(self, upload_row_frame):
        # Gesture selection frame (right side)
        gesture_frame = ctk.CTkFrame(upload_row_frame, fg_color="#E8ECEF", corner_radius=8)
        gesture_frame.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")

        # Configure frame layout
        gesture_frame.grid_columnconfigure(0, weight=1)
        gesture_frame.grid_rowconfigure(0, weight=0)  # Title
        gesture_frame.grid_rowconfigure(1, weight=0)  # Start gesture section
        gesture_frame.grid_rowconfigure(2, weight=0)  # Space
        gesture_frame.grid_rowconfigure(3, weight=0)  # End gesture section

        # Title
        title_label = ctk.CTkLabel(
            gesture_frame,
            text="Gesture Selection",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="center"
        )
        title_label.grid(row=0, column=0, padx=10, pady=(15, 10), sticky="ew")

        # Gesture options
        gesture_options = ["Peace", "OK", "Open", "Close"]

        # Load icons
        self.gesture_icons = {}
        for gesture in gesture_options:
            icon_path = f"graphics/hand_signs/{gesture}.png"
            try:
                # Load the image with PIL
                original_img = Image.open(icon_path)
                self.gesture_pil_images[gesture] = original_img

                # Create initial PhotoImage at default size
                target_width = 24
                target_height = 24
                resized_img = original_img.resize((target_width, target_height), Image.LANCZOS)
                self.gesture_icons[gesture] = ImageTk.PhotoImage(resized_img)
            except Exception:
                # If it fails, we'll use text as fallback
                messagebox.showwarning("Warning", f"Cannot load hand icon: {gesture}")
                self.gesture_pil_images[gesture] = None
                self.gesture_icons[gesture] = None

        # Start gesture section
        start_section = ctk.CTkFrame(gesture_frame, fg_color="transparent")
        start_section.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        start_section.grid_columnconfigure(0, weight=1)

        start_label = ctk.CTkLabel(
            start_section,
            text="Start Gesture",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        start_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")

        # Create custom combobox with icons for start gesture
        self.create_icon_combobox(start_section, self.start, gesture_options, 1)

        # Space
        spacer = ctk.CTkFrame(gesture_frame, fg_color="transparent", height=10)
        spacer.grid(row=2, column=0, sticky="ew")

        # End gesture section
        end_section = ctk.CTkFrame(gesture_frame, fg_color="transparent")
        end_section.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        end_section.grid_columnconfigure(0, weight=1)

        end_label = ctk.CTkLabel(
            end_section,
            text="End Gesture",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        end_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")

        # Create custom combobox with icons for end gesture
        self.create_icon_combobox(end_section, self.end, gesture_options, 1)

    def create_icon_combobox(self, parent, variable, options, row):
        # Create a frame to hold the selection display and dropdown button
        combo_frame = ctk.CTkFrame(parent, fg_color="white", corner_radius=6, height=40)
        combo_frame.grid(row=row, column=0, padx=5, pady=(5, 15), sticky="ew")
        combo_frame.grid_columnconfigure(0, weight=1)  # Icon+text area
        combo_frame.grid_columnconfigure(1, weight=0)  # Button

        # Initial selected item
        selected_item = variable.get() if variable.get() in options else options[0]
        variable.set(selected_item)

        combo_frame.icon_photo = None

        # Selected item display (with icon if available)
        if self.gesture_icons.get(selected_item):
            # Define your desired size
            target_width = 24
            target_height = 24

            # Create a new PhotoImage at the desired size
            resized_img = self.gesture_pil_images[selected_item].resize((target_width, target_height), Image.LANCZOS)
            combo_frame.icon_photo = ImageTk.PhotoImage(resized_img)

            icon_label = tk.Label(
                combo_frame,
                image=combo_frame.icon_photo,
                background="white"
            )
            icon_label.grid(row=0, column=0, padx=(5, 0), pady=5, sticky="w")

            text_label = ctk.CTkLabel(
                combo_frame,
                text=selected_item,
                anchor="w",
                fg_color="transparent"
            )
            text_label.grid(row=0, column=0, padx=(40, 0), pady=5, sticky="w")
        else:
            text_label = ctk.CTkLabel(
                combo_frame,
                text=selected_item,
                anchor="w",
                fg_color="transparent"
            )
            text_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Dropdown arrow button
        dropdown_btn = ctk.CTkButton(
            combo_frame,
            text="▼",
            width=30,
            height=30,
            fg_color="transparent",
            text_color="gray",
            hover_color="#F0F0F0",
            command=lambda: self.show_dropdown_menu(parent, combo_frame, variable, options, row)
        )
        dropdown_btn.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="e")

    def show_dropdown_menu(self, parent, combo_frame, variable, options, row):
        # Create dropdown menu
        menu = tk.Menu(parent, tearoff=0)

        # Add the options
        for option in options:
            if self.gesture_icons.get(option):
                menu.add_command(
                    label=f" {option}",
                    command=lambda opt=option: self.update_combobox(combo_frame, variable, opt)
                )
            else:
                menu.add_command(
                    label=option,
                    command=lambda opt=option: self.update_combobox(combo_frame, variable, opt)
                )

        # Position and display the menu
        x = combo_frame.winfo_rootx()
        y = combo_frame.winfo_rooty() + combo_frame.winfo_height()
        menu.post(x, y)

    def update_combobox(self, combo_frame, variable, selected_option):
        # Update the variable
        variable.set(selected_option)

        # Destroy all children in column 0
        for widget in combo_frame.grid_slaves(row=0, column=0):
            widget.destroy()

        # Update the display
        if self.gesture_icons.get(selected_option):
            icon_label = tk.Label(
                combo_frame,
                image=self.gesture_icons[selected_option],
                background="white"
            )
            icon_label.grid(row=0, column=0, padx=(5, 0), pady=5, sticky="w")

            text_label = ctk.CTkLabel(
                combo_frame,
                text=selected_option,
                anchor="w",
                fg_color="transparent"
            )
            text_label.grid(row=0, column=0, padx=(40, 0), pady=5, sticky="w")
        else:
            text_label = ctk.CTkLabel(
                combo_frame,
                text=selected_option,
                anchor="w",
                fg_color="transparent"
            )
            text_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    def _setup_loading_frame(self):
        """Set up the loading screen for processing indication."""
        # Configure frame layout
        self.loading_frame.grid_columnconfigure(0, weight=1)
        self.loading_frame.grid_rowconfigure(0, weight=1)

        # Loading elements
        loading_label = ctk.CTkLabel(
            self.loading_frame,
            text="Processing your video...",
            font=ctk.CTkFont(size=20),
            text_color="#606770"
        )
        loading_label.grid(row=0, column=0, padx=20, pady=20)

        # Loading progress bar (centered)
        progress = ctk.CTkProgressBar(self.loading_frame, width=300)
        progress.grid(row=1, column=0, padx=20, pady=20)
        progress.configure(mode="indeterminate")
        progress.start()

    def _setup_results_frame(self):
        """Set up the results frame to display processed video clips."""
        # Configure frame layout
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=0)  # Header
        self.results_frame.grid_rowconfigure(1, weight=1)  # Video grid
        self.results_frame.grid_rowconfigure(2, weight=0)  # Back button

        # Header for results (centered)
        results_header = ctk.CTkFrame(self.results_frame, fg_color="white", corner_radius=8)
        results_header.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="ew")

        # Configure header for centering
        results_header.grid_columnconfigure(0, weight=1)

        # Results title
        results_label = ctk.CTkLabel(
            results_header,
            text="Your Short Videos",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#606770"
        )
        results_label.grid(row=0, column=0, padx=20, pady=10)

        # Scrollable frame for videos (centered)
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.results_frame,
            fg_color="#E8ECEF",
            corner_radius=8
        )
        self.scrollable_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # Back button (centered)
        back_button = ctk.CTkButton(
            self.results_frame,
            text="Back to Home",
            command=lambda: self.show_frame(self.home_frame),
            width=150,
            font=ctk.CTkFont(weight="bold")
        )
        back_button.grid(row=2, column=0, padx=20, pady=20)

    def show_frame(self, frame):
        """Bring the specified frame to the front.

        Args:
            frame: The frame to display.
        """
        frame.tkraise()

    def select_video(self):
        """Open file dialog to select a video file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.MOV")]
        )
        if file_path:
            self.selected_file.set(file_path)

    def select_custom_sign(self, sign_type):
        """Open file dialog to select a custom sign image and save it as PNG to preserve transparency."""
        file_path = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("GIF files", "*.gif"), ("All Image Files", "*.png;*.gif;*.jpg;*.jpeg")]
        )

        if not file_path:
            return

        try:
            # Check if 'signs' directory exists, create it if it doesn't
            if not os.path.exists(ref_signs_folder):
                os.makedirs(ref_signs_folder)

            # Destination path (always save as PNG)
            dest_path = os.path.join(ref_signs_folder, sign_filename)

            # Copy the image file to the signs directory
            import shutil
            shutil.copy(file_path, dest_path)

            # Update the appropriate variable
            if sign_type == "start":
                self.custom_start_image = dest_path
            self._update_sign_display(self.start_sign_display, self.start_sign_widget, dest_path, sign_type)
            self.sign_detector.initialize()

            print(f"Custom {sign_type} sign image saved to: {dest_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save custom sign image: {str(e)}")

    def _update_sign_display(self, display_frame, widget, image_path, sign_type):
        """Update the sign display with the selected image and add an X button.

        Args:
            display_frame: The frame containing the image display
            widget: The current widget in the display (to be replaced)
            image_path: Path to the image file
            sign_type: 'start' or 'end' to specify which sign
        """
        # Clear existing widgets
        if widget:
            widget.destroy()

        try:
            # Create a frame to hold both the image and X button
            container = ctk.CTkFrame(display_frame, fg_color="#E8ECEF", corner_radius=0)
            container.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)

            # Load and display the image using PIL and CTkImage
            pil_image = Image.open(image_path)

            # Resize image to fit the display
            max_size = 200  # Slightly smaller than the 150px container
            img_width, img_height = pil_image.size
            scale = min(max_size / img_width, max_size / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image,
                                     size=(new_width, new_height))

            # Create label to display the image
            img_label = ctk.CTkLabel(container, image=ctk_image, text="")
            img_label.place(relx=0.5, rely=0.5, anchor="center")

            # Add X button to remove the image
            x_button = ctk.CTkButton(
                container,
                text="✕",
                width=20,
                height=20,
                fg_color="red",
                hover_color="darkred",
                command=lambda: self._remove_sign_image(sign_type)
            )
            x_button.place(relx=0.9, rely=0.1, anchor="ne")

            # Update the reference to the new widget
            if sign_type == "start":
                self.start_sign_widget = container
            else:
                self.end_sign_widget = container

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display sign image: {str(e)}")
            # Reset to no image
            label = ctk.CTkLabel(display_frame, text="Error loading image", text_color="red")
            label.place(relx=0.5, rely=0.5, anchor="center")
            if sign_type == "start":
                self.start_sign_widget = label
                self.custom_start_image = None

    def _remove_sign_image(self, sign_type):
        """Remove the custom sign image and related files with enhanced debugging.

        Args:
            sign_type: 'start' or 'end' to specify which sign to remove
        """
        try:

            # Clear the image from memory and reset display
            if sign_type == "start":
                self.custom_start_image = None
                # Reset display
                self.start_sign_widget.destroy()
                self.start_sign_widget = ctk.CTkLabel(
                    self.start_sign_display,
                    text="No image selected",
                    text_color="gray"
                )
                self.start_sign_widget.place(relx=0.5, rely=0.5, anchor="center")

            # Function to safely delete files with logging
            def _safe_delete(path, description):
                if path:  # Check if path is not None or empty
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            return True  # Indicate successful deletion
                        except OSError as e:
                            messagebox.showerror("Error", f"Failed to remove {description} {path}: {e}")
                            return False  # Indicate failure
                    else:
                        print(f"Warning: {description} {path} does not exist.")
                        return True  # Indicate that it is already deleted
                else:
                    print(f"Info: {description} path is None/empty, skipping deletion.")
                    return True  # Nothing to delete, so consider it "successful"

            # Delete files
            _safe_delete(sign_path, "sign image")
            _safe_delete(model_path, "model")
            _safe_delete(keypoint_path, "keypoint")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove sign image: {str(e)}")

    def handle_drop(self, event):
        """Handle drag-and-drop event for video files."""
        try:
            # Get the dropped file path
            file_path = event.data
            # Clean up the file path (remove curly braces if multiple files are dropped)
            if file_path.startswith('{') and file_path.endswith('}'):
                file_path = file_path[1:-1].split()[0]  # Take the first file if multiple
            # Check if the file is a valid video file
            if Path(file_path).suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                self.selected_file.set(file_path)
            else:
                messagebox.showwarning("Invalid File", "Please drop a valid video file (.mp4, .avi, .mkv, .mov)!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the dropped file: {str(e)}")

    def execute_split(self):
        from ELS import run_main_in_thread
        """Process the video and show results."""
        if not self.selected_file.get():
            messagebox.showwarning("No File", "Please select a video first!")
            return
        if self.mode.get() == "custom_sign":
            if not self.sign_detector.initialize():
                messagebox.showwarning("No Sign Found", "No custom sign found! Aborting video processing...")
                return
        # self.sign_detector.run_camera_detection()
        self.show_frame(self.loading_frame)
        run_main_in_thread(self)

    def show_results(self):
        """Display the results screen with processed videos."""
        self.populate_videos()
        self.show_frame(self.results_frame)

    def generate_thumbnail(self, video_path, max_size=(180, 120)):
        """Generate a proportional thumbnail image from a random frame in the video."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return None

            import random
            frame_number = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret:
                return None

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Resize while keeping aspect ratio
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

            cap.release()
            return pil_image
        except Exception as e:
            print(f"Error generating thumbnail for {video_path}: {e}")
            return None

    def populate_videos(self):
        """Create video thumbnail entries from real videos in the results screen."""
        # Clear existing widgets in scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Get the folder path with real videos
        folder_path = f'videos/{self.date_time.get()}'

        # Check if folder exists
        if not os.path.exists(folder_path):
            # Show no videos message
            no_videos_label = ctk.CTkLabel(
                self.scrollable_frame,
                text="No videos found in the specified folder",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            no_videos_label.pack(pady=50)
            return

        # Get list of video files
        video_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]

        if not video_files:
            # Show no videos message
            no_videos_label = ctk.CTkLabel(
                self.scrollable_frame,
                text="No video files found in the folder",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            no_videos_label.pack(pady=50)
            return

        # Create a 3-column grid for videos
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)

        # Create video entries
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(folder_path, video_file)

            # Get video metadata if possible (optional)
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Convert to MB

            # Create frame for this video
            video_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="white", corner_radius=8)
            video_frame.grid(row=i // 3, column=i % 3, padx=10, pady=10, sticky="nsew")

            # Configure video frame for centering content
            video_frame.grid_columnconfigure(0, weight=1)

            # Generate and display thumbnail
            thumbnail_image = self.generate_thumbnail(video_path)
            if thumbnail_image:
                # Use actual image size to preserve aspect ratio
                ctk_image = ctk.CTkImage(
                    light_image=thumbnail_image,
                    dark_image=thumbnail_image,
                    size=thumbnail_image.size
                )
                thumbnail = ctk.CTkLabel(
                    video_frame,
                    image=ctk_image,
                    text="",
                    corner_radius=4
                )
            else:
                # Fallback placeholder if thumbnail generation fails
                thumbnail = ctk.CTkLabel(
                    video_frame,
                    text="[Thumbnail Error]",
                    fg_color="#E8ECEF",
                    text_color="#606770",
                    width=180,
                    height=120,
                    corner_radius=4
                )
            thumbnail.grid(row=0, column=0, padx=10, pady=(20, 10), sticky="ew")

            # Video title (actual filename)
            title = ctk.CTkLabel(
                video_frame,
                text=video_file,
                font=ctk.CTkFont(size=12),
                text_color="#1A73E8",
                wraplength=180  # Wrap long filenames
            )
            title.grid(row=1, column=0, padx=10, pady=(0, 5))

            # File size info
            file_info = ctk.CTkLabel(
                video_frame,
                text=f"{file_size:.1f} MB",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            file_info.grid(row=2, column=0, padx=10, pady=(0, 5))

            # Button frame (centered)
            button_frame = ctk.CTkFrame(video_frame, fg_color="white", corner_radius=0)
            button_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

            # Center the buttons within the frame
            button_frame.grid_columnconfigure(0, weight=1)
            button_frame.grid_columnconfigure(1, weight=1)
            button_frame.grid_columnconfigure(2, weight=1)

            # Play button
            play_btn = ctk.CTkButton(
                button_frame,
                text="Play",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25,
                command=lambda v=video_path: self.play_video(v)
            )
            play_btn.grid(row=0, column=0, padx=2, pady=5)

            # Open folder button
            folder_btn = ctk.CTkButton(
                button_frame,
                text="Folder",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25,
                command=lambda f=folder_path: self.open_folder(f)
            )
            folder_btn.grid(row=0, column=1, padx=2, pady=5)

            # More options button
            more_btn = ctk.CTkButton(
                button_frame,
                text="•••",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25,
                command=lambda v=video_path: self.show_options(v)
            )
            more_btn.grid(row=0, column=2, padx=2, pady=5)

    def play_video(self, video_path):
        """Open the video in the default video player"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(video_path)
            elif os.name == 'posix':  # macOS, Linux
                import subprocess
                subprocess.call(('open', video_path) if sys.platform == 'darwin' else ('xdg-open', video_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video: {e}")

    def open_folder(self, folder_path):
        """Open the folder containing the videos"""
        folder_path = folder_path.rsplit('/', 1)[0]
        try:
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            elif os.name == 'posix':  # macOS, Linux
                import subprocess
                subprocess.call(('open', folder_path) if sys.platform == 'darwin' else ('xdg-open', folder_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")

    def show_options(self, video_path):
        """Show additional options for the video"""
        options = ["Rename", "Delete", "Copy path"]

        # Create a simple popup menu
        popup = tk.Menu(self.master, tearoff=0)
        popup.add_command(label="Rename", command=lambda: self.rename_video(video_path))
        popup.add_command(label="Delete", command=lambda: self.delete_video(video_path))
        popup.add_command(label="Copy path", command=lambda: self.copy_path(video_path))

        # Display the popup menu
        try:
            popup.tk_popup(self.master.winfo_pointerx(), self.master.winfo_pointery())
        finally:
            popup.grab_release()

    def rename_video(self, video_path):
        """Rename the video file while preserving its extension"""
        old_name = os.path.basename(video_path)
        file_name, file_ext = os.path.splitext(old_name)

        # Ask user for the new name (without extension)
        new_name_without_ext = simpledialog.askstring("Rename", "Enter new name:",
                                                      initialvalue=file_name)

        if new_name_without_ext and new_name_without_ext != file_name:
            try:
                # Create the new full name by combining the new name with the original extension
                new_full_name = new_name_without_ext + file_ext
                new_path = os.path.join(os.path.dirname(video_path), new_full_name)

                os.rename(video_path, new_path)
                self.populate_videos()  # Refresh the view
            except Exception as e:
                messagebox.showerror("Error", f"Could not rename file: {e}")

    def delete_video(self, video_path):
        """Delete the video file"""
        if messagebox.askyesno("Confirm Delete",
                               f"Are you sure you want to delete {os.path.basename(video_path)}?"):
            try:
                os.remove(video_path)
                self.populate_videos()  # Refresh the view
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete file: {e}")

    def copy_path(self, video_path):
        """Copy the video path to clipboard"""
        self.master.clipboard_clear()
        self.master.clipboard_append(video_path)
        messagebox.showinfo("Info", "Path copied to clipboard")

    def check_for_saved_signs(self):
        """Check for existing sign images and display them if found"""
        if not os.path.exists(ref_signs_folder):
            return

        # Check for start sign
        sign_files = [f for f in os.listdir(ref_signs_folder) if f.startswith(sign_filename)]
        if sign_files:
            path_to_sign = os.path.join(ref_signs_folder, sign_files[0])
            self.custom_start_image = path_to_sign
            self._update_sign_display(self.start_sign_display, self.start_sign_widget, path_to_sign, "start")
            self.sign_detector.initialize()
