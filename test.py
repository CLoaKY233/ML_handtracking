import cv2
import numpy as np
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

        # Apply the skin color filter
        filtered_frame = apply_skin_color_filter(frame)

        # Convert the filtered frame to grayscale
        gray_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)

        # Save each frame as an image in the output folder
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, gray_frame)

        # Display the grayscale frame
        cv2.imshow("Grayscale Frame", gray_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    message_label.config(text="Recording complete. Grayscale images saved in the selected folder.")

# Function to apply the skin color filter
def apply_skin_color_filter(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a color range for detecting skin color (you may need to adjust these values)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    # Create a mask to detect the skin color
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Use the skin mask to make the hand region white in the original frame
    white_hand = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # Create a darkened background
    dark_background = np.zeros_like(frame)

    # Combine the white_hand and dark_background to create the final frame
    final_frame = cv2.add(white_hand, dark_background)

    return final_frame

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
