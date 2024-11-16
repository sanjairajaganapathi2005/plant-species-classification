from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

model = load_model("dcnn_model_ps.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        
        img = Image.open(io.BytesIO(f.read()))
        img = img.resize((100, 100))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        y = model.predict(x)
        preds = np.argmax(y, axis=1)

        index = [
    "Alstonia Scholaris diseased",
    "Alstonia Scholaris healthy",
    "Arjun diseased",
    "Arjun healthy",
    "Bael diseased",
    "Basil healthy",
    "Chinar diseased",
    "Chinar healthy",
    "Guava diseased",
    "Guava healthy",
    "Jamun diseased",
    "Jamun healthy",
    "Jatropha diseased",
    "Jatropha healthy",
    "Lemon diseased",
    "Lemon healthy",
    "Mango diseased",
    "Mango healthy",
    "Pomegranate diseased",
    "Pomegranate healthy",
    "Pongamia Pinnata diseased",
    "Pongamia Pinnata healthy"
]

        text = "The classified plant specials is: /n" + str(index[preds[0]])
        return text

if __name__ == '__main__':
    app.run(debug=True,port=5000)

