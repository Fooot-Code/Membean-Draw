import cv2
import numpy as np
import pyautogui
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.datasets import mnist
from emnist import extract_training_samples, extract_test_samples

def preprocess_drawing(canvas):
    # Convert to grayscale and resize to 28x28 (standard size for MNIST)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    # Normalize pixel values
    normalized = resized / 255.0
    # Reshape for model input
    return normalized.reshape(1, 28, 28, 1)

def load_recognition_model():
    # Check if model exists, if not, train it
    if not os.path.exists('handwriting_model.keras'):
        print("Please train the model first using the EMNIST dataset")
        return None
    return tf.keras.models.load_model('handwriting_model.keras')

def detect_drawing():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Create a blank canvas for drawing
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_point = None
    drawing = False
    drawn_points = []
    
    # Load the recognition model
    model = load_recognition_model()
    if model is None:
        return
        
    # Map predictions to characters
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define blue color range (assuming blue marker/object)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            if M["m00"] > 0:
                # Get center point
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                if not drawing:
                    drawing = True
                    prev_point = (cx, cy)
                    drawn_points = [(cx, cy)]
                else:
                    # Draw line on canvas
                    cv2.line(canvas, prev_point, (cx, cy), (255, 255, 255), 2)
                    prev_point = (cx, cy)
                    drawn_points.append((cx, cy))
        else:
            if drawing:
                # Drawing finished, process the drawn shape
                if len(drawn_points) > 10:  # Minimum points for a letter
                    # Preprocess the drawing
                    processed_image = preprocess_drawing(canvas)
                    
                    # Get prediction from model
                    prediction = model.predict(processed_image)
                    predicted_char = characters[np.argmax(prediction)]
                    
                    # Type the predicted character
                    pyautogui.write(predicted_char)
                    
                    print(f"Recognized character: {predicted_char}")
                
                # Reset for next drawing
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                drawing = False
                drawn_points = []
        
        # Combine frame and canvas
        combined = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
        
        # Show the frame
        cv2.imshow('Draw Letters', combined)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def train_model():
    """
    Function to train the handwriting recognition model using EMNIST dataset.
    This should be run once before using the main program.
    """
    # Load EMNIST dataset (letters)
    x_train, y_train = extract_training_samples('letters')
    x_test, y_test = extract_test_samples('letters')
    
    # Normalize and reshape the data
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    # Adjust labels to 0-25 range (EMNIST letters dataset uses 1-26)
    y_train = y_train - 1
    y_test = y_test - 1
    
    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(26, activation='softmax')  # 26 letters only
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    # Save the model
    model.save('handwriting_model.keras')

if __name__ == "__main__":
    # Check if model exists, if not train it
    if not os.path.exists('handwriting_model.keras'):
        print("Training new model...")
        train_model()
    
    # Add small delay before starting
    time.sleep(2)
    detect_drawing()

