from flask import Blueprint, request, jsonify

summerisizer = Blueprint('summerisizer', __name__)

@summerisizer.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return jsonify({"status": "ok"}), 200