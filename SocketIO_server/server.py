import socketio

# Create a Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

@sio.event
def connect(sid, environ):
    print('Client connected:', sid)

@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)

@sio.on('predict')
def handle_predict(sid, data):
    print('Received prediction request:', data)
    # Broadcast the prediction to all connected clients
    sio.emit('prediction', data)

if __name__ == '__main__':
    import eventlet
    import eventlet.wsgi

    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

