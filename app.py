from flask import Flask, render_template, redirect, request, url_for
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.applications.resnet_v2 import preprocess_input
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

cottonNET_model = load_model('YOUR_MODEL_PATH')

class PredictingPipeline():
    def __init__(self):
        pass

    def preprocess_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        # Normalizing the image array
        img_array = img_array.astype('float32') / 255.0
        # (224, 224, 3) -> (1, 224, 224, 3) [batch]
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def predict(self, model_input):
        preds = cottonNET_model.predict(model_input)
        # Getting the index of the maximum probability chance
        preds = np.argmax(preds, axis=1)
        return preds
        

## Flask App

app = Flask(__name__)

app.secret_key = "supersecretkey"  # Set your own secret key
app.config["UPLOAD_FOLDER"] = "uploads"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files["file"]
        # Check if a file was selected
        if file.filename == "":
            return redirect(url_for("predict"))
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        # Using the PredictingPipeline to predict
        pipeline = PredictingPipeline()
        preprocess_image = pipeline.preprocess_image(img_path=file_path)
        preds = pipeline.predict(preprocess_image)
        return render_template('predict.html', prediction=preds)
    else:
        return render_template('predict.html', prediction='')

if __name__ == '__main__':
    app.run(debug=True)