import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the MediaPipe hands module
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get the screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize index finger position
index_x = 0
index_y = 0

# Create a Tkinter window
root = tk.Tk()
root.title("Virtual Mouse")

# Create a Canvas to display the webcam feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Define a function to exit the application
def exit_app():
    cap.release()  # Release the webcam
    root.destroy()  # Close the application

# Create an exit button
exit_button = tk.Button(root, text="Exit", command=exit_app, bd=0, bg="red", fg="white", font=("Arial", 12))
exit_button.pack(side=tk.BOTTOM, padx=20, pady=20)

# Define a function to update the virtual mouse
def update():
    global index_x, index_y

    # Capture a frame from the webcam
    _, frame = cap.read()
    frame = cv2.flip (frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe hands
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:
                    # Update index finger position
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    index_x = screen_width * x / frame_width
                    index_y = screen_height * y / frame_height

                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    thumb_x = screen_width * x / frame_width
                    thumb_y = screen_height * y / frame_height

                    # Check for thumb and index finger proximity
                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click()
                        pyautogui.sleep(1)
                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y)

    # Display the frame on the canvas
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)

    canvas.create_image(0, 0, anchor='nw', image=img)
    canvas.image = img

    # Schedule the next frame update
    root.after(10, update)

# Start updating the frames
update()

# Run the Tkinter main loop
root.mainloop()
