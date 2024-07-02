from flask import Flask, render_template
import socketio

app = Flask(__name__)

sio = socketio.Client()

# Connect to the socket server
sio.connect('http://localhost:5000')

@app.route('/')
def index():
    return render_template('index.html')

@sio.on('prediction')
def on_prediction(data):
    print('Received prediction:', data)
    # You can store the prediction and render it on the webpage

if __name__ == '__main__':
    app.run(port=5002, debug=True)

