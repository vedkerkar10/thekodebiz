import json
import os
import signal
import threading
from module.server_clean import app
from module.htmlServer import SimpleHTTPServer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def stop_server(signum, frame):
    print("Stopping server...")
    os._exit(0)  # Force exit all threads.

if __name__ == '__main__':
    signal.signal(signal.SIGINT, stop_server)  # Handle Ctrl+C
    with open('server_details.json') as f:
        json_data = json.load(f)
    
    server_details = (json_data['server_ip'], json_data['server_port'])
    http_server = SimpleHTTPServer(server_details=server_details, web_directory='public/out')
    
    # Start the HTTP server in a separate thread
    thread = threading.Thread(target=http_server.start, daemon=True)
    thread.start()

    # Run Flask app
    app.run(debug=True, host="0.0.0.0")
