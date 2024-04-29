from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('injury_detection.h5')

# Function to preprocess a single frame for inference
def preprocess_frame(frame, target_size=(150, 150)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = cv2.resize(frame, target_size)  # Resize frame to target size
    return frame

# Function to make predictions on a single frame
def predict_frame(frame):
    # Reshape frame for model input
    input_data_reshaped = frame.reshape(-1, 150, 150, 3)
    # Make predictions using the loaded model
    predictions = model.predict(input_data_reshaped)
    # Convert predictions to binary classes (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int)
    return predicted_classes

# Function to generate video frames with predictions
def generate_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        # if not ret:
        #     break
        processed_frame = preprocess_frame(frame)
        predicted_classes = predict_frame(processed_frame)
        if predicted_classes[0] == 0:
            text = "Minor Face Injury"
        else:
            text = "Major Face Injury"

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
