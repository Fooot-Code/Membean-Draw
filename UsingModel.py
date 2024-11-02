import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import pyautogui
import time

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        # Create canvas
        self.canvas = Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack(pady=20)
        
        # Create clear button
        clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        clear_button.pack(pady=5)
        
        # Create predict button
        predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        predict_button.pack(pady=5)
        
        # Initialize drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Bind predict to 'p' key
        self.root.bind('p', lambda event: self.predict_digit())
        
        # Load model
        self.model = tf.keras.models.load_model('handwriting_model.keras')

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x = event.x
            y = event.y
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                     fill='white', width=20, capstyle=tk.ROUND, 
                                     smooth=True)
            self.last_x = x
            self.last_y = y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict_digit(self):
        time.sleep(1)

        # Get the canvas content
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # Create PIL image from canvas
        image = Image.new('L', (280, 280), color='black')
        draw = ImageDraw.Draw(image)
        
        # Copy canvas contents to image
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            draw.line(coords, fill='white', width=20)
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array
        digit_array = np.array(image)
        
        # Normalize the array
        digit_array = digit_array.astype('float32') / 255.0
        
        # Reshape for model prediction
        digit_array = digit_array.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = self.model.predict(digit_array)
        predicted_index = np.argmax(prediction[0])
        
        # Convert index to corresponding letter (assuming A-Z mapping)
        predicted_letter = chr(65 + predicted_index)  # 65 is ASCII for 'A'
        print(f"Predicted letter: {predicted_letter}")

        # Type out the predicted letter using pyautogui
        pyautogui.write(predicted_letter.lower())

def main():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
