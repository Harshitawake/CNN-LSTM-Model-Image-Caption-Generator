from flask import Flask, render_template, request, send_from_directory
import os
import socketio
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from pickle import load

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load tokenizer and model
tokenizer = load(open("C://Users/harsh/Desktop/image_caption_generator/tokenizer.p", "rb"))
model = load_model('C://Users/harsh/Desktop/image_caption_generator/models/model_28.keras')
xception_model = Xception(include_top=False, pooling="avg")

sio = socketio.Client()

# Connect to the socket server
sio.connect('http://localhost:5000')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_path = request.form['image_path']
        if not os.path.exists(image_path):
            return render_template('index.html', error='File not found. Please provide a valid image path.')

        filename = os.path.basename(image_path)
        new_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        try:
            with open(image_path, 'rb') as f:
                with open(new_path, 'wb') as new_file:
                    new_file.write(f.read())
            photo = extract_features(image_path, xception_model)
            caption = generate_caption(model, tokenizer, photo, 32)
            sio.emit('predict', {'image_path': new_path, 'caption': caption})
            return render_template('index.html', image_path=new_path, caption=caption)
        except Exception as e:
            return render_template('index.html', error=f'Error copying file: {str(e)}')

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def extract_features(image_path, model):
    image = preprocess_image(image_path)
    features = model.predict(image)
    return features

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

if __name__ == '__main__':
    app.run(port=5001, debug=True)
