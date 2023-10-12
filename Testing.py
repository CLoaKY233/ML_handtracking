import cv2
import numpy as np
import tensorflow as tf

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

# Load the palm detection model
model = tf.keras.models.load_model('palm_detection_model.h5')

# Function to detect palms using the ML model
def detect_palm(frame):
    # Preprocess the frame as needed for your model
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match your model's input shape
    input_shape = (128, 128)  # Adjust this to match your model's input shape
    frame_resized = cv2.resize(frame_gray, (input_shape[1], input_shape[0]))

    # Predict using the model
    predictions = model.predict(np.expand_dims(frame_resized, axis=0))  # Assuming your model expects a batch of images

    # Process the predictions to identify palm locations
    # This part is specific to your model's output format and post-processing

    # Example: If your model outputs class probabilities and you want to detect palms based on a threshold
    threshold = 0.5 # Adjust this threshold as needed
    detected_palm_frame = frame.copy()  # Make a copy of the frame

    if predictions[0][0] >= threshold:
        # Palm detected, you can draw a rectangle or mark it in some way
        cv2.rectangle(detected_palm_frame, (0, 0), (input_shape[1], input_shape[0]), (0, 255, 0), 2)

    return detected_palm_frame

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply the skin color filter
    filtered_frame = apply_skin_color_filter(frame)

    # Detect palms using your ML model
    detected_palm_frame = detect_palm(filtered_frame)

    # Display the frame with palms detected
    cv2.imshow("Palm Detection", detected_palm_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
