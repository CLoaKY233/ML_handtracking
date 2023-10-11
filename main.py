import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Specify the path to the dataset folder
dataset_dir = r'C:\Users\laysh\OneDrive\Desktop\dataset_folder\leapGestRecog\03\01_palm'

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
model.save("palm_detection_model.keras")
