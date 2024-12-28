from flask import Blueprint, jsonify, request
from ..models import Combination, db
import csv
from io import StringIO

api_bp = Blueprint('api', __name__)

@api_bp.route('/add_combination', methods=['POST'])
def add_combination():
    data = request.get_json()
    numbers = data.get('numbers')
    special = data.get('special')
    if not numbers or special is None:
        return jsonify({"error": "Invalid data"}), 400
    
    combination = Combination(numbers=numbers, special=special)
    db.session.add(combination)
    db.session.commit()
    return jsonify({"message": "Combination added successfully"}), 201

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
