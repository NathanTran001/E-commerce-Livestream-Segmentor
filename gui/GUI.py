import tkinter as tk
from tkinter import filedialog, messagebox

# Main application window
root = tk.Tk()
root.title("Video to Multiple Short Videos")
root.geometry("400x300")
root.resizable(False, False)

# Function to switch between frames (screens)
def show_frame(frame):
    frame.tkraise()

# Create three frames for different states
input_frame = tk.Frame(root)
loading_frame = tk.Frame(root)
results_frame = tk.Frame(root)

# Stack frames on top of each other
for frame in (input_frame, loading_frame, results_frame):
    frame.grid(row=0, column=0, sticky="nsew")

# --- Input Frame (Initial Screen) ---
input_label = tk.Label(input_frame, text="Select a video to split", font=("Arial", 14))
input_label.pack(pady=20)

selected_file = tk.StringVar()
file_label = tk.Label(input_frame, textvariable=selected_file, wraplength=350)
file_label.pack(pady=10)

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if file_path:
        selected_file.set(f"Selected: {file_path.split('/')[-1]}")

select_button = tk.Button(input_frame, text="Select Video", command=select_video, width=15)
select_button.pack(pady=10)

def execute_split():
    if not selected_file.get():
        messagebox.showwarning("No File", "Please select a video first!")
        return
    show_frame(loading_frame)
    # Simulate execution with a delay (replace with your cutting logic)
    root.after(2000, show_results)  # 2-second delay for demo

execute_button = tk.Button(input_frame, text="Execute", command=execute_split, width=15)
execute_button.pack(pady=20)

# --- Loading Frame ---
loading_label = tk.Label(loading_frame, text="Processing your video...", font=("Arial", 16))
loading_label.pack(expand=True)

# --- Results Frame ---
results_label = tk.Label(results_frame, text="Your Short Videos", font=("Arial", 14))
results_label.pack(pady=10)

# Scrollable listbox to display cut video names (mockup)
scrollbar = tk.Scrollbar(results_frame)
video_list = tk.Listbox(results_frame, height=10, width=50, yscrollcommand=scrollbar.set)
scrollbar.config(command=video_list.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
video_list.pack(pady=10)

def show_results():
    # Mockup: Populate with example video names (replace with real output)
    video_list.delete(0, tk.END)
    for i in range(1, 6):
        video_list.insert(tk.END, f"short_video_{i}.mp4")
    show_frame(results_frame)

back_button = tk.Button(results_frame, text="Back", command=lambda: show_frame(input_frame), width=15)
back_button.pack(pady=10)

# Start with the input frame visible
show_frame(input_frame)

# Run the application
root.mainloop()