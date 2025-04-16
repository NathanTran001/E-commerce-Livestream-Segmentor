import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import *
from pathlib import Path


def main():
    """Main entry point for the Video to Multiple Short Videos application."""
    # Set appearance mode and default color theme for CustomTkinter
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    # Main application window with DnD support
    root = TkinterDnD.Tk()
    root.title("Video to Multiple Short Videos")
    root.geometry("800x600")
    root.resizable(True, True)

    # Create App instance and run it
    app = VideoSplitterApp(root)
    root.mainloop()


class VideoSplitterApp:
    """Application for splitting videos into multiple shorter clips.

    This class provides a CustomTkinter-based GUI for uploading videos
    and splitting them into shorter segments.
    """

    def __init__(self, master):
        """Initialize the application.

        Args:
            master: The root Tk window of the application.
        """
        self.master = master
        self.selected_file = tk.StringVar()
        self.start_gesture = tk.StringVar(value="Select Start Gesture")
        self.end_gesture = tk.StringVar(value="Select End Gesture")

        # Set up frames for different screens
        self._setup_frames()

        # Start with the input frame visible
        self.show_frame(self.input_frame)

    def _setup_frames(self):
        """Set up the main frames for different application states."""
        # Create three frames for different states with the same grid position
        self.input_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)
        self.loading_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)
        self.results_frame = ctk.CTkFrame(self.master, fg_color="#E8ECEF", corner_radius=0)

        # Stack frames on top of each other
        for frame in (self.input_frame, self.loading_frame, self.results_frame):
            frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid to allow expansion
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Set up each frame's content
        self._setup_input_frame()
        self._setup_loading_frame()
        self._setup_results_frame()

    def _setup_input_frame(self):
        """Set up the input frame with upload functionality and gesture selection."""
        # Configure frame layout
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_rowconfigure(0, weight=0)  # Header
        self.input_frame.grid_rowconfigure(1, weight=1)  # Drag area
        self.input_frame.grid_rowconfigure(2, weight=0)  # File label
        self.input_frame.grid_rowconfigure(3, weight=0)  # Gesture selection
        self.input_frame.grid_rowconfigure(4, weight=0)  # Execute button
        self.input_frame.grid_rowconfigure(5, weight=0)  # Footer

        # Header frame (centered)
        header_frame = ctk.CTkFrame(self.input_frame, fg_color="white", corner_radius=8)
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="ew")

        # Configure header for centering
        header_frame.grid_columnconfigure(0, weight=1)

        # Upload button (centered in header frame)
        upload_button = ctk.CTkButton(
            header_frame,
            text="Upload Video",
            command=self.select_video,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        upload_button.grid(row=0, column=0, padx=20, pady=10)

        # Drag-and-drop area
        self.drag_area = ctk.CTkFrame(self.input_frame, fg_color="white", corner_radius=8)
        self.drag_area.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # Configure drag area for centering
        self.drag_area.grid_rowconfigure(0, weight=1)
        self.drag_area.grid_rowconfigure(1, weight=0)
        self.drag_area.grid_rowconfigure(2, weight=1)
        self.drag_area.grid_columnconfigure(0, weight=1)

        # Upload icon (centered)
        upload_icon = ctk.CTkLabel(
            self.drag_area,
            text="⬆",
            font=ctk.CTkFont(size=40),
            text_color="#A9A9A9"
        )
        upload_icon.grid(row=0, column=0, padx=20, pady=20)

        # Text labels (centered)
        drag_label = ctk.CTkLabel(
            self.drag_area,
            text="Select video to upload",
            font=ctk.CTkFont(size=16)
        )
        drag_label.grid(row=1, column=0, padx=20, pady=(0, 5))

        drag_sublabel = ctk.CTkLabel(
            self.drag_area,
            text="Or drag and drop video files",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        drag_sublabel.grid(row=2, column=0, padx=20, pady=(0, 20))

        # Enable drag-and-drop using tkinterdnd2
        self.drag_area.drop_target_register(DND_FILES)
        self.drag_area.dnd_bind('<<Drop>>', self.handle_drop)

        # Selected file label
        file_label = ctk.CTkLabel(
            self.input_frame,
            textvariable=self.selected_file,
            wraplength=700
        )
        file_label.grid(row=2, column=0, padx=20, pady=5)

        # Gesture selection frame
        gesture_frame = ctk.CTkFrame(self.input_frame, fg_color="#E8ECEF", corner_radius=0)
        gesture_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        gesture_frame.grid_columnconfigure(0, weight=1)
        gesture_frame.grid_columnconfigure(1, weight=1)

        # Gesture options
        gesture_options = ["Peace ✌️", "OK 👌", "Open 👐", "Close ✊"]

        # Start gesture dropdown
        start_gesture_label = ctk.CTkLabel(
            gesture_frame,
            text="Start Gesture:",
            font=ctk.CTkFont(size=14)
        )
        start_gesture_label.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="e")

        start_gesture_menu = ctk.CTkOptionMenu(
            gesture_frame,
            variable=self.start_gesture,
            values=gesture_options,
            width=150
        )
        start_gesture_menu.grid(row=0, column=1, padx=(0, 20), pady=5, sticky="w")

        # End gesture dropdown
        end_gesture_label = ctk.CTkLabel(
            gesture_frame,
            text="End Gesture:",
            font=ctk.CTkFont(size=14)
        )
        end_gesture_label.grid(row=1, column=0, padx=(0, 10), pady=5, sticky="e")

        end_gesture_menu = ctk.CTkOptionMenu(
            gesture_frame,
            variable=self.end_gesture,
            values=gesture_options,
            width=150
        )
        end_gesture_menu.grid(row=1, column=1, padx=(0, 20), pady=5, sticky="w")

        # Execute button (centered)
        execute_button = ctk.CTkButton(
            self.input_frame,
            text="Execute",
            command=self.execute_split,
            width=150,
            font=ctk.CTkFont(weight="bold")
        )
        execute_button.grid(row=4, column=0, padx=20, pady=10)

        # Footer/info section (centered)
        footer_frame = ctk.CTkFrame(self.input_frame, fg_color="#E8ECEF", corner_radius=0)
        footer_frame.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="ew")

        # Configure footer for centering
        footer_frame.grid_columnconfigure(0, weight=1)

        # Version info
        info_label = ctk.CTkLabel(
            footer_frame,
            text="Video to Multiple Short Videos App v1.0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        info_label.grid(row=0, column=0, padx=20, pady=(5, 0))

        # Description
        desc_label = ctk.CTkLabel(
            footer_frame,
            text="Easily split your videos into shorter clips!",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc_label.grid(row=1, column=0, padx=20, pady=(0, 5))

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
            text="Back",
            command=lambda: self.show_frame(self.input_frame),
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
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if file_path:
            self._set_selected_file(file_path)

    def handle_drop(self, event):
        """Handle drag-and-drop event for video files."""
        try:
            # Get the dropped file path
            file_path = event.data
            # Clean up the file path (remove curly braces if multiple files are dropped)
            if file_path.startswith('{') and file_path.endswith('}'):
                file_path = file_path[1:-1].split()[0]  # Take the first file if multiple
            # Check if the file is a valid video file
            if Path(file_path).suffix.lower() in ['.mp4', '.avi', '.mkv']:
                self._set_selected_file(file_path)
            else:
                messagebox.showwarning("Invalid File", "Please drop a valid video file (.mp4, .avi, .mkv)!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the dropped file: {str(e)}")

    def _set_selected_file(self, file_path):
        """Set the selected file and update the UI."""
        filename = Path(file_path).name
        self.selected_file.set(f"Selected: {filename}")

    def execute_split(self):
        """Process the video and show results."""
        if not self.selected_file.get():
            messagebox.showwarning("No File", "Please select a video first!")
            return
        if self.start_gesture.get() == "Select Start Gesture" or self.end_gesture.get() == "Select End Gesture":
            messagebox.showwarning("No Gesture", "Please select both start and end gestures!")
            return

        self.show_frame(self.loading_frame)

        # Simulate processing delay (2 seconds)
        self.master.after(2000, self.show_results)

    def show_results(self):
        """Display the results screen with processed videos."""
        self.populate_videos()
        self.show_frame(self.results_frame)

    def populate_videos(self):
        """Create example video thumbnail entries in the results screen."""
        # Clear existing widgets in scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Create a 3-column grid for videos
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)

        # Create mock video thumbnails
        for i in range(5):  # Mock 5 videos
            video_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="white", corner_radius=8)
            video_frame.grid(row=i // 3, column=i % 3, padx=10, pady=10, sticky="nsew")

            # Configure video frame for centering content
            video_frame.grid_columnconfigure(0, weight=1)

            # Placeholder for video thumbnail
            thumbnail = ctk.CTkLabel(
                video_frame,
                text="[Video Thumbnail]",
                fg_color="#E8ECEF",
                text_color="#606770",
                width=180,
                height=120,
                corner_radius=4
            )
            thumbnail.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

            # Video title
            title = ctk.CTkLabel(
                video_frame,
                text=f"short_video_{i + 1}.mp4",
                font=ctk.CTkFont(size=12),
                text_color="#1A73E8"
            )
            title.grid(row=1, column=0, padx=10, pady=(0, 5))

            # Mock views
            views = ctk.CTkLabel(
                video_frame,
                text="0 views",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            views.grid(row=2, column=0, padx=10, pady=(0, 5))

            # Button frame (centered)
            button_frame = ctk.CTkFrame(video_frame, fg_color="white", corner_radius=0)
            button_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

            # Center the buttons within the frame
            button_frame.grid_columnconfigure(0, weight=1)
            button_frame.grid_columnconfigure(1, weight=1)
            button_frame.grid_columnconfigure(2, weight=1)

            # Action buttons
            embed_btn = ctk.CTkButton(
                button_frame,
                text="</>",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25
            )
            embed_btn.grid(row=0, column=0, padx=2, pady=5)

            edit_btn = ctk.CTkButton(
                button_frame,
                text="Edit",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25
            )
            edit_btn.grid(row=0, column=1, padx=2, pady=5)

            more_btn = ctk.CTkButton(
                button_frame,
                text="•••",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25
            )
            more_btn.grid(row=0, column=2, padx=2, pady=5)


if __name__ == "__main__":
    main()