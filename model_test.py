import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from pickle import load

# Argument parser setup
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

# Load and preprocess the image
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Can't open image! Ensure that image path and extension are correct.")
        return None
    image = image.resize((299, 299))
    image = np.array(image)
    # for 4 channels images, we need to convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

# Function to map an integer to a word in the tokenizer
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        #print(sequence)
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        print(pred)
        word = word_for_id(pred, tokenizer)
        print(word)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Max length of the sequence
max_length = 32

# Load tokenizer and model
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('models/model_28.keras')

# Load Xception model for feature extraction
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the input image
photo = extract_features(img_path, xception_model)
print(photo)

# Generate description for the image
if photo is not None:
    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    print(description)

    # Display the image
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
else:
    print("Feature extraction failed.")
