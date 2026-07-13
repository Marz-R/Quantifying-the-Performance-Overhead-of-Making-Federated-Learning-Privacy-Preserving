from flask import Flask, request, jsonify
from waitress import serve
from bulletin import Public_Bulletin

MAX_EPOCH = 100

app = Flask(__name__)
bulletin = Public_Bulletin({}, num_epochs=MAX_EPOCH)


@app.route("/add_peer", methods=["POST"])
def add_peer():
    data = request.json
    bulletin.add_peer(data["peer_id"], data["peer_info"])
    return jsonify({"status": "ok"})


@app.route("/remove_peer", methods=["POST"])
def remove_peer():
    bulletin.remove_peer(request.json["peer_id"])
    return jsonify({"status": "ok"})


@app.route("/get_peer_list", methods=["GET"])
def get_peer_list():
    return jsonify({"version": bulletin.version, "peer_list": bulletin.get_peer_list()})


@app.route("/peer_list/since/<int:version>", methods=["GET"])
def update_peer_list(version):
    if bulletin.version > version:
        return jsonify({"version": bulletin.version, "peer_list": bulletin.get_peer_list()})
    else:
        return jsonify({"version": version, "peer_list": None})  # nothing changed


@app.route("/get_num_epochs", methods=["GET"])
def get_num_epochs():
    return jsonify({"num_epochs": bulletin.get_num_epochs()})


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)