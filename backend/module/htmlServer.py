import http.server
import socketserver
import threading
import socket

class SimpleHTTPServer:
    def __init__(self, server_details, web_directory):
        self.server_details = server_details
        self.web_directory = web_directory
        self.server = None
        self.thread = None

    def start(self):
        handler = self._make_handler()
        max_attempts = 10
        current_port = self.server_details[1]

        for attempt in range(max_attempts):
            try:
                # Try to create the server with the current port
                self.server = socketserver.TCPServer(
                    (self.server_details[0], current_port), 
                    handler
                )
                self.server.allow_reuse_address = True  # Allow port reuse
                break
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    if attempt == max_attempts - 1:
                        raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")
                    current_port += 1
                else:
                    raise

        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        print(f"Server running at http://{self.server_details[0]}:{current_port}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
            print("Server stopped.")

    def _make_handler(self):
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=self.server_instance.web_directory, **kwargs)

        CustomHandler.server_instance = self
        return CustomHandler

if __name__ == "__main__":
    # Create an instance of SimpleHTTPServer
    http_server = SimpleHTTPServer(('localhost', 49155), 'public/out')
    
    # Start the server
    try:
        http_server.start()
    except KeyboardInterrupt:
        http_server.stop()