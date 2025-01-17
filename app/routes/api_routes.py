
## app/routes/api_routes.py

from io import StringIO
from flask import Blueprint, request, render_template, jsonify
from app.models import Combination
from app.utils.prediction_utils import predict_next_combination
from datetime import datetime
from flask_cors import CORS
import csv
import joblib
from app import db 

api_bp = Blueprint('api', __name__)

@api_bp.route('/add_combination', methods=['POST'])
def add_combination():
    print('Entra al método add_combination')
    data = request.get_json()
    numbers = data.get('numbers')
    special = data.get('special')
    
    # Validar datos de entrada
    if not numbers or len(numbers) != 5 or not special:
        return jsonify({'success': False, 'error': 'Datos inválidos'}), 400

    try:
        numbers = [int(num) for num in numbers]
        special = int(special)

        # Validar unicidad de los números
        if len(set(numbers)) != 5:
            return jsonify({'success': False, 'error': 'Los números no deben repetirse.'}), 400

        # Validar rango de números
        if not all(1 <= num <= 43 for num in numbers):
            return jsonify({'success': False, 'error': 'Los números deben estar entre 1 y 43.'}), 400

        # Validar rango del número especial
        if not (1 <= special <= 16):
            return jsonify({'success': False, 'error': 'El número especial debe estar entre 1 y 16.'}), 400

    except ValueError:
        return jsonify({'success': False, 'error': 'Error al convertir números.'}), 400

    # Guardar en la base de datos
    combination = Combination(numbers=','.join(map(str, numbers)), special=special)
    db.session.add(combination)
    db.session.commit()

    print(f"Combinación añadida: {combination}")  # Mensaje de depuración

    # Entrenar y guardar el modelo después de añadir la nueva combinación
    #train_and_save_model()

    return jsonify({'success': True, 'message': 'Combination added successfully'}), 201

@api_bp.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
    reader = csv.reader(stream)
    for row in reader:
        numbers, special = row[0], int(row[1])
        combination = Combination(numbers=numbers, special=special)
        db.session.add(combination)
    db.session.commit()
    return jsonify({"message": "CSV data uploaded successfully"}), 201

@api_bp.route('/get_combinations', methods=['GET'])
def get_combinations():
    combinations = Combination.query.all()
    data = [{"id": c.id, "numbers": c.numbers, "special": c.special} for c in combinations]
    return jsonify(data)


## Método que returna la predicción
@api_bp.route('/predict', methods=['GET'])
def predict():
    combinations = [
        {'numbers': list(map(int, comb.numbers.split(','))), 'special': comb.special}
        for comb in Combination.query.all()
    ]
    next_numbers, next_special = predict_next_combination(combinations)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response = {
        "estado": "exitoso",
        "fecha_hora": timestamp,
        "mensaje": "La predicción se ha generado con éxito.",
        "prediccion": {
            "balotas": next_numbers,
            "super_balota": next_special
        },
    }
    return jsonify(response)

