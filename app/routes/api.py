from flask import Blueprint, request, jsonify
from app.models import Combination, db
from app.services.prediction_service import train_and_save_model

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/add_combination', methods=['POST'])
def add_combination():
    # Lógica para agregar combinaciones.
    pass

@api_bp.route('/upload_csv', methods=['POST'])
def upload_csv():
    # Lógica para cargar combinaciones desde CSV.
    pass
