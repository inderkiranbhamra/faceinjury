# import cv2
# from flask import Flask, render_template, Response
# from tensorflow.keras.models import load_model
# import numpy as np
#
# app = Flask(__name__)
#
# # Load the trained model
# model = load_model('injury_detection.h5')
#
# # OpenCV video capture
# camera = cv2.VideoCapture(1)
#
#
# def preprocess_image(frame):
#     # Resize and normalize the frame
#     frame = cv2.resize(frame, (150, 150))
#     frame = frame / 255.0
#     return frame.reshape(-1, 150, 150, 3)
#
#
# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             # Perform inference on the frame
#             preprocessed_frame = preprocess_image(frame)
#             prediction = model.predict(preprocessed_frame)
#             predicted_class = "Heavily Injured" if prediction == 1 else "Minor Damage"
#
#             # Draw prediction text on the frame
#             cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#             # Encode the frame as JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#
#             # Yield the frame as part of the video stream
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')  # render the HTML template
#
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')  # return the response generated along with the specific media type
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
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
        cv2.putText(frame, "Predicted Class: {}".format(predicted_classes[0]), (10, 30),
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