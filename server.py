from flask import Flask
from threading import Thread
from server_app import app_func

thread = Thread(target = app_func)

app = Flask(__name__)

@app.route('/')
def server_alive():
    if not thread.is_alive():
        thread.start()
    else:
        pass
    return 'server is alive and running...!!!'

if __name__ == "__main__":
    app.run(debug=True)