from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from keras_preprocessing import image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

gestures = ['paper', 'rock', 'scissors']

app = Flask(__name__)

print("Loading model")
model = tf.keras.models.load_model('rps.h5')


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    img = image.load_img(os.path.join('uploads', filename), target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    probabilities = model.predict([x])[0, :]
    index = np.argsort(probabilities)
    print(gestures[index[2]])
    predictions = {
        "class1": gestures[index[2]],
        "class2": gestures[index[1]],
        "class3": gestures[index[0]],
        "prob1": probabilities[index[2]],
        "prob2": probabilities[index[1]],
        "prob3": probabilities[index[0]],
    }
    return render_template('predict.html', predictions=predictions)


# app.run(host='0.0.0.0')
