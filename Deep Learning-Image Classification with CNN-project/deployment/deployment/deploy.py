from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

# Setting Up the Flask App
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Loading the model
model = tf.keras.models.load_model("C:/Users/ASUS/OneDrive/Desktop/ironhacklab/week_three/project/project two/skipconnection 89_model.h5")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Image Preprocessing 
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Ensure image is in RGB mode
    image = image.resize((32, 32))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Serving the HTML File
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Defining the Prediction API to handle multiple images
@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400  # Handle missing files

    files = request.files.getlist('files')
    predictions_list = []
    
    try:
        for file in files[:4]:  # Limit to 4 images
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = class_names[np.argmax(predictions[0])]  # Get the highest probability class
            probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}  # Convert to JSON-serializable format
            
            predictions_list.append({
                'predicted_class': predicted_class,
                'probabilities': probabilities
            })
        
        return jsonify({'predictions': predictions_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle errors during processing

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run Flask app
