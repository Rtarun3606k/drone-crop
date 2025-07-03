from flask import Blueprint


Model = Blueprint('Model',__name__)

@Model.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify if the server is running.
    """
    return {"status": "ok"}, 200


