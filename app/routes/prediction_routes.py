# app/routes/prediction_routes.py

from flask import Blueprint, jsonify
from app.utils.prediction_utils import generate_prediction

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['GET'])
def predict():
    response = generate_prediction()
    return jsonify(response)
