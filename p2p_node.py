import socket
import struct
import threading
import pickle
from typing import Dict, Tuple

HEADER_SIZE = 4 # bytes of the length prefix in front of every packet

class Node(threading.Thread):
    def __init__(self, host, port, id, debug_message = False):
        super(Node, self).__init__()
        self.host = host
        self.port = port
        self.id = str(id)

        self.connected_nodes: Dict[str, Tuple[socket.socket, Tuple[str, int]]] = {} # id -> [socket, (host, port)]

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._lock = threading.Lock()
        self.terminate_flag = threading.Event()

        self.debug_message = debug_message


    def print_debug_messages(self, message):
        if self.debug_message:
            print("**DEBUG** Node " + self.id + " : " + message)


    def init_server(self):
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.port = self.sock.getsockname()[1]
        self.sock.settimeout(1.0)
        self.sock.listen()
        self.print_debug_messages("Node listening on " + str(self.host) + ":" + str(self.port))


    def stop(self):
        self.terminate_flag.set()
        self.sock.close()
        for peer_id in list(self.connected_nodes.keys()):
            self.disconnect_with_node(peer_id)
        self.print_debug_messages("Node stopped.")


    def run(self): # threading.start() will call this method
        self.init_server()
        self._listen_for_connections()


    def _listen_for_connections(self):
        while not self.terminate_flag.is_set():
            try:
                conn, addr = self.sock.accept()

            except socket.timeout:
                continue # checks for terminate_flag every second

            except OSError:
                break 

            threading.Thread(target=self._accept_handshake, args=(conn, ), daemon=True).start()


    def _accept_handshake(self, conn: socket.socket):
        try:
            payload = self._recv_packet(conn)
            if payload is None:
                self.print_debug_messages("Failed to receive handshake from peer. Closing connection.")
                conn.close()
                return
            
            peer_info = pickle.loads(payload)
            self._send_handshake(conn)

        except (OSError, pickle.PickleError):
            self.print_debug_messages("Failed to process handshake from peer. Closing connection.")
            conn.close()
            return

        self._register_connection(peer_info["id"], conn, peer_info["host"], peer_info["port"])


    def _register_connection(self, peer_id: str, conn: socket.socket, peer_host: str, peer_port: int):
        with self._lock:
            if peer_id in self.connected_nodes:
                self.print_debug_messages("Already connected to node " + str(peer_id) + ". Closing new connection.")
                conn.close()
                return
            
            self.connected_nodes[peer_id] = (conn, (peer_host, peer_port))
            self.print_debug_messages("Connected to node " + str(peer_id) + " at " + str(peer_host) + ":" + str(peer_port))

        threading.Thread(target=self._listen_for_messages, args=(peer_id, conn,), daemon=True).start()


    def connect_with_node(self, peer_host: str, peer_port: int):
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((peer_host, peer_port))

        self._send_handshake(conn)

        try:
            payload = self._recv_packet(conn)
            if payload is None:
                self.print_debug_messages("Failed to receive handshake from peer. Closing connection.")
                conn.close()
                return
            
            peer_info = pickle.loads(payload)

        except (OSError, pickle.PickleError):
            self.print_debug_messages("Failed to process handshake from peer. Closing connection.")
            conn.close()
            return

        self._register_connection(peer_info["id"], conn, peer_info["host"], peer_info["port"])


    def _send_handshake(self, conn: socket.socket):        
        payload = pickle.dumps({"id": self.id, "host": self.host, "port": self.port})
        conn.sendall(struct.pack('>I', len(payload)) + payload)


    def disconnect_with_node(self, peer_id: str):
        with self._lock:
            peer_conn = self.connected_nodes.get(peer_id)
        if peer_conn is None:
            self.print_debug_messages("Cannot disconnect from node " + str(peer_id) + " because it is not connected.")
            self._remove_connection(peer_id)
            return

        conn, _ = peer_conn
        try:
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()

        except OSError:
            pass

        self._remove_connection(peer_id)


    def _remove_connection(self, peer_id: str):
        if peer_id in self.connected_nodes:
            with self._lock:
                self.connected_nodes.pop(peer_id, None)


    def _listen_for_messages(self, peer_id: str, conn: socket.socket):
        while not self.terminate_flag.is_set():
            try:
                payload = self._recv_packet(conn)
                if payload is not None:
                    message = pickle.loads(payload)
                    self.node_message(peer_id, message, len(payload) + HEADER_SIZE)

            except (OSError, pickle.PickleError):
                self.print_debug_messages("Failed to receive message from node " + str(peer_id) + ". Disconnecting.")
                break

        self.disconnect_with_node(peer_id)


    def send_to_node(self, peer_id: str, message): # returns the number of bytes sent
        with self._lock:
            peer_conn = self.connected_nodes.get(peer_id)
        if peer_conn is None:
            self.print_debug_messages("Cannot send message to node " + str(peer_id) + " because it is not connected.")
            self._remove_connection(peer_id)
            return 0

        conn, _ = peer_conn
        try:
            return self._send_packet(conn, message)

        except (OSError, pickle.PickleError):
            self.print_debug_messages("Failed to send message to node " + str(peer_id) + ". Disconnecting.")
            self.disconnect_with_node(peer_id)
            return 0


    def node_message(self, peer_id: str, message, num_bytes: int = 0):
        pass  # Overridden in PeerNode


    @staticmethod
    def _recv_bytes(conn: socket.socket, size: int):
        data = bytearray()

        while len(data) < size:
            chunk = conn.recv(size - len(data))
            if not chunk:
                return None
            
            data.extend(chunk)

        return bytes(data)

    @staticmethod
    def _recv_packet(conn: socket.socket):
        header = Node._recv_bytes(conn, HEADER_SIZE)
        if not header:
            return None

        message_length = struct.unpack("!I", header)[0]
        payload = Node._recv_bytes(conn, message_length)

        return payload

    @staticmethod
    def _send_packet(conn: socket.socket, packet): # returns the number of bytes sent
        payload = pickle.dumps(packet)
        header = struct.pack('!I', len(payload))
        conn.sendall(header + payload)

        return len(header) + len(payload)