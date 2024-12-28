from flask import Blueprint, jsonify
from app.services.prediction_service import predict_next_combination
from app.models import Combination

predictions_bp = Blueprint('predictions', __name__, url_prefix='/predictions')

@predictions_bp.route('/predict', methods=['GET'])
def predict():
    # Lógica para realizar predicciones.
    pass
