import os
import sys
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from tkinterdnd2 import *

start_sign = "Select Start Gesture"
end_sign = "Select End Gesture"

class VideoSplitterApp:
    """Application for splitting videos into multiple shorter clips.

    This class provides a CustomTkinter-based GUI for uploading videos
    and splitting them into shorter segments.
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
        # Set up frames for different screens
        self._setup_frames()

        # Start with the input frame visible
        self.show_frame(self.input_frame)
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
        gesture_options = ["Peace", "OK", "Open", "Close"]

        # Start gesture dropdown
        start_gesture_label = ctk.CTkLabel(
            gesture_frame,
            text="Start Gesture:",
            font=ctk.CTkFont(size=14)
        )
        start_gesture_label.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="e")

        start_gesture_menu = ctk.CTkOptionMenu(
            gesture_frame,
            variable=self.start,
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
            variable=self.end,
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
            self.selected_file.set(file_path)

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
                self.selected_file.set(file_path)
            else:
                messagebox.showwarning("Invalid File", "Please drop a valid video file (.mp4, .avi, .mkv)!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the dropped file: {str(e)}")

    def execute_split(self):
        from ELS import run_main_in_thread
        """Process the video and show results."""
        if not self.selected_file.get():
            messagebox.showwarning("No File", "Please select a video first!")
            return
        if start_sign == "Select Start Gesture":
            messagebox.showwarning("No Gesture", "Please select start gesture!")
            return

        self.show_frame(self.loading_frame)
        run_main_in_thread(self)

    def show_results(self):
        """Display the results screen with processed videos."""
        self.populate_videos()
        self.show_frame(self.results_frame)

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

            # Placeholder for video thumbnail (still using placeholder since generating
            # actual thumbnails requires additional processing)
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

            # Play button - replace embed button
            play_btn = ctk.CTkButton(
                button_frame,
                text="Play",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25,
                command=lambda v=video_path: self.play_video(v)
            )
            play_btn.grid(row=0, column=0, padx=2, pady=5)

            # Open folder button - replace edit button
            folder_btn = ctk.CTkButton(
                button_frame,
                text="Folder",
                font=ctk.CTkFont(size=10),
                width=50,
                height=25,
                command=lambda f=folder_path: self.open_folder(self.selected_file.get())
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
        popup.add_command(label="Copy path", command=lambda: self.copy_path(self.selected_file.get()))

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


