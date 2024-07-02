# CNN-LSTM Image Caption Generator

This project implements an image captioning generator using a Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) network. The model takes an image as input and generates a descriptive caption.

## Table of Contents

- [Introduction](#introduction)
- [Approach](#approach)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [Combining CNN and LSTM](#combining-cnn-and-lstm)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Introduction

Image captioning is the task of generating a textual description for a given image. It involves understanding the objects in the image, their attributes, and how they interact. This project uses a combination of CNN and LSTM to perform this task.

## Approach

### Convolutional Neural Network (CNN)

A CNN is used to extract features from the input images. These features represent various aspects of the image, such as edges, textures, and objects. The CNN used in this project is pre-trained on a large dataset (e.g., VGG16, InceptionV3) and fine-tuned for our specific task.

### Long Short-Term Memory (LSTM)

An LSTM network is a type of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies. In this project, the LSTM network is used to generate sequences of words based on the features extracted by the CNN.

### Combining CNN and LSTM

The CNN extracts features from the image, and these features are passed to the LSTM, which generates the caption word by word. The architecture can be summarized as follows:

1. **Image Feature Extraction:** A pre-trained CNN (e.g., VGG16, InceptionV3, Xception) is used to extract a fixed-length vector from the image.
2. **Caption Generation:** The extracted features are fed into an LSTM network, which generates the caption.

## Dataset

The model is trained on a dataset of images and corresponding captions. Popular datasets for image captioning include:

- MS COCO (Microsoft Common Objects in Context)
- Flickr8k
- Flickr30k

Each image in the dataset has multiple captions describing it from different perspectives.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV
- Flask (for the web application)
- Flask-SocketIO (for real-time caption broadcasting)
- Pickle (for loading the tokenizer)

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Harshitawake/CNN-LSTM-Model-Image-Caption-Generator.git
   cd CNN-LSTM-Image-Caption-Generator

2. **Start the SocketIO server:**
    ```bash
    cd SocketIO_server
    python server.py

3. **Start the flask app1 for prediction on web page app:**
    ```bash
    cd ..
    cda app1
    python app.py

4. **Start the flask app2 for Broadcasting the prediction in real time on 2nd web page app:**
    ```bash
    cd ..
    cda app2
    python app.py

