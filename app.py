from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from uuid import uuid4
from nutrition_details import get_fruit_nutrition_details

# Initialize Flask app
app = Flask(__name__)
CORS(app, origin='0.0.0.0', headers=['Content- Type', 'Authorization'])
# CORS(app, origin='localhost', headers=['Content- Type', 'Authorization'])

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to handle file uploads and return nutrition details
@app.route('/api/get_nutrition_details', methods=['POST'])
def get_nutrition_details():
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({"error": "No image part in the request"}), 400

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            filename = secure_filename(str(uuid4()) + '_' + file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the file as needed
            nutrition = get_fruit_nutrition_details(filepath)
            return nutrition, 200
    except Exception as e:
        return jsonify({"error": e}), 400


app.run(host='0.0.0.0', port=5000, debug=True)
# app.run(host='localhost', port=5000, debug=True)
