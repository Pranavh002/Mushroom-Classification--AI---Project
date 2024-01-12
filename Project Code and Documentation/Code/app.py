import numpy as np
import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

model = load_model("mushroom.h5")
app = Flask(__name__)

# Create the 'uploads' folder if it doesn't exist
uploads_dir = os.path.join(app.root_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/input')
def input1():
    return render_template("input.html")

@app.route('/new_page')
def new_page():
    return render_template("new_page.html")

@app.route('/images')
def images():
    return render_template("images.html")

@app.route('/predict', methods=["POST"])
def predict():
    f = request.files['image']
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(uploads_dir, f.filename)
    f.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    img_data = preprocess_input(x)
    prediction = np.argmax(model.predict(img_data), axis=1)

    index = ['"Boletus"', '"Lactarius"', '"Russula"']
    result = index[prediction[0]]

    return render_template('output.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
