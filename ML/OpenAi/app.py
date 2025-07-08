from flask import Flask, jsonify, request
from flask_cors import CORS




app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify if the server is running.
    """
    return jsonify({"status": "ok"}), 200



