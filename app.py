# app.py

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import pickle

app = Flask(__name__)

# Load pickle model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Preprocess uploaded image
def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert image to numpy array
    image_array = np.array(image)

    # Invert colors (optional for MNIST style)
    image_array = 255 - image_array

    # Normalize pixel values
    image_array = image_array / 255.0

    # Flatten image
    image_array = image_array.flatten()

    # Reshape for prediction
    image_array = image_array.reshape(1, -1)

    return image_array


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        # Check file upload
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        try:
            # Open image
            image = Image.open(file)

            # Preprocess image
            processed_image = preprocess_image(image)

            # Predict digit
            prediction = np.argmax(model.predict(processed_image), axis=1)[0]

        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)