from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load your trained Keras model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def model_predict(img_path, model,class_names):
    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name[2:], confidence_score

@app.route('/')
def index():
    # Main page
    return 'API Server Started'

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Ensure the 'uploads' folder exists
        uploads_folder = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)

        # Save the file to ./uploads
        file_path = os.path.join(uploads_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        class_name, confidence_score = model_predict(file_path, model,class_names)
        print(class_name,len(class_name))
        return class_name

if __name__ == '__main__':
    app.run(port=5001, debug=True)