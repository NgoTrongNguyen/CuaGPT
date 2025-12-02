# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
app.config['SECRET_KEY'] = 'skibidi bop yes yes'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
def handle_send_message(data):
    # data = {'username': '...', 'message': '...', 'room': 'optional'}
    room = data.get('room')
    if room:
        emit('receive_message', data, room=room)
    else:
        emit('receive_message', data, broadcast=True)

@socketio.on('join')
def on_join(data):
    # data = {'username': '...', 'room': 'room-name'}
    username = data['username']
    room = data['room']
    join_room(room)
    emit('receive_message', {'username': 'System', 'message': f'{username} đã vào phòng {room}'}, room=room)

@socketio.on('leave')
def on_leave(data):
    username = data['username']
    room = data['room']
    leave_room(room)
    emit('receive_message', {'username': 'System', 'message': f'{username} đã rời phòng {room}'}, room=room)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)