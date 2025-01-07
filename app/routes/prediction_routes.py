# app/routes/prediction_routes.py

from flask import Blueprint, jsonify
from app.models import Combination
from app.utils.prediction_utils import predict_next_combination
from datetime import datetime

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['GET'])
def predict():
    # Obtener combinaciones de la base de datos
    combinations = [
        {'numbers': list(map(int, comb.numbers.split(','))), 'special': comb.special}
        for comb in Combination.query.all()
    ]
    
    # Pasar combinaciones a la función
    next_numbers, next_special = predict_next_combination(combinations)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    """ response = {
        "predicted_numbers": next_numbers,
        "predicted_special": next_special
    } """
    
    response = {
        "estado": "exitoso",  # Indicador de éxito
        "fecha_hora": timestamp,  # Fecha y hora actual
        "mensaje": "La predicción se ha generado con éxito. Aquí están los números recomendados para la próxima jugada:",
        "informacion_adicional": "Estas son las características clave que el modelo utilizó para hacer la predicción. El modelo usó 100 estimadores para generar esta predicción.",
        "prediccion": {
            "balotas": next_numbers,
            "super_balota": next_special
        },    
    }
    
    print("Respuesta de predicción: ",response)
    
    return jsonify(response)
