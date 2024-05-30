from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# ArUcoのライブラリを導入
aruco = cv2.aruco

# 4x4のマーカー、IDは50までの辞書を使用
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        marker_ids = recognize_ar_marker(filepath)
        return jsonify({"detected_ids": marker_ids})
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello", 200

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def recognize_ar_marker(image_path):
    img = cv2.imread(image_path)
    corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=parameters)
    if ids is not None:
        return ids.ravel().tolist()
    else:
        return []

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
