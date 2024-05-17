import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Function to start capturing and saving images
def start_capture():
    global frame_count
    global output_folder

    # Ensure the output folder is selected
    if not output_folder:
        message_label.config(text="Please select an output folder.")
        return

    message_label.config(text="Recording in progress...")

    # Open a connection to the webcam (usually 0 or 1, depending on your setup)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save each frame as an image in the output folder
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Display the captured frame
        cv2.imshow("Captured Frame", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    message_label.config(text="Recording complete. Images saved in the selected folder.")

# Function to select the output folder
def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if output_folder:
        message_label.config(text=f"Output folder selected: {output_folder}")
    else:
        message_label.config(text="No output folder selected.")

# Initialize the frame count and output folder
frame_count = 0
output_folder = None

# Create the main tkinter window
root = tk.Tk()
root.title("Image Capture")

# Create a label to display messages
message_label = tk.Label(root, text="", padx=10, pady=10)
message_label.pack()

# Create buttons for starting and stopping the capture and for selecting the output folder
start_button = tk.Button(root, text="Start Capture", command=start_capture)
select_folder_button = tk.Button(root, text="Select Output Folder", command=select_output_folder)
quit_button = tk.Button(root, text="Quit", command=root.quit)

start_button.pack()
select_folder_button.pack()
quit_button.pack()

# Start the tkinter main loop
root.mainloop()
