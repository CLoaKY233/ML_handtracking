import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox

train_button = None
dataset_dir = None  # Store the selected dataset directory

# Function to load and preprocess images
def load_and_preprocess_data(dataset_dir):
    image_paths = []
    labels = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))
            image = image / 255.0  # Normalize pixel values to [0, 1]

            # Assign a label based on the folder name (01_palm or not)
            label = 1 if "01_palm" in dataset_dir else 0
            image_paths.append(image)
            labels.append(label)

    return image_paths, labels

# Function to train the model
def train_model():
    global dataset_dir  # Use the global dataset_dir

    if dataset_dir is None:
        messagebox.showerror("Error", "Please select a dataset folder.")
        return

    # Load and preprocess data
    image_paths, labels = load_and_preprocess_data(dataset_dir)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Define a simple CNN model for palm detection
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)  # Fit the data generator to the training data

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=10)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy}')

    # Save and use the model for palm detection
    model.save("palm_detection_model.h5")
    messagebox.showinfo("Training Complete", "Model trained successfully and saved as 'palm_detection_model.h5'.")

# Function to select the dataset folder
def select_dataset_folder():
    global dataset_dir
    dataset_dir = filedialog.askdirectory(title="Select Dataset Folder")
    if dataset_dir:
        train_button.state(['!disabled'])
        train_button["text"] = "Train Model for:\n" + dataset_dir

# Create a simple Tkinter GUI for starting the process
root = tk.Tk()
root.title("Palm Detection Trainer")

frame = ttk.Frame(root)
frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

label = ttk.Label(frame, text="Welcome to Palm Detection Trainer!")
label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

select_button = ttk.Button(frame, text="Select Dataset Folder", command=select_dataset_folder)
select_button.grid(row=1, column=0, padx=10, pady=10)

train_button = ttk.Button(frame, text="Train Model", command=train_model)
train_button.grid(row=1, column=1, padx=10, pady=10)
train_button.state(['disabled'])

root.mainloop()
