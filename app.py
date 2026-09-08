from flask import Flask, request, jsonify, render_template
import os
import uuid
from bubble_counter import count_bubbles

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    filename = f"{uuid.uuid4()}.mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    count = count_bubbles(filepath)
    return jsonify({"bubble_count": count})


if __name__ == "__main__":
    app.run(debug=True)