 
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import random
import csv
from collections import defaultdict

app = Flask(__name__)

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://pablo:P4bl03011@localhost/combinations_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializar la base de datos
db = SQLAlchemy(app)

# Permitir solo el origen específico
CORS(app, resources={r"/add_combination": {"origins": "http://127.0.0.1:5500"},
                     r"/upload_csv": {"origins": "http://127.0.0.1:5500"}})

# Modelo de la base de datos para almacenar combinaciones
class Combination(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    numbers = db.Column(db.String(100), nullable=False)
    special = db.Column(db.Integer, nullable=False)

    def __init__(self, numbers, special):
        self.numbers = ','.join(map(str, numbers))
        self.special = special

# Modelo para el historial de predicciones
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    numbers = db.Column(db.String(100), nullable=False)
    special = db.Column(db.Integer, nullable=False)

    def __init__(self, numbers, special):
        self.numbers = ','.join(map(str, numbers))
        self.special = special

# Crear todas las tablas necesarias
with app.app_context():
    db.create_all()

# Endpoint para agregar una combinación
@app.route('/add_combination', methods=['POST'])
def add_combination():
    data = request.get_json()

    numbers = data.get('numbers')
    special = data.get('special')

    if not numbers or len(numbers) != 5 or not special:
        return jsonify({'success': False, 'error': 'Datos inválidos'})

    try:
        numbers = [int(num) for num in numbers]
        special = int(special)
    except ValueError:
        return jsonify({'success': False, 'error': 'Error al convertir números'})

    # Guardar en la base de datos
    combination = Combination(numbers=numbers, special=special)
    db.session.add(combination)
    db.session.commit()

    print(f"Combinación añadida: {combination}")  # Mensaje de depuración

    return jsonify({'success': True})

# Endpoint para cargar combinaciones desde un CSV
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csvFile' not in request.files:
        return jsonify({'success': False, 'error': 'No se encontró el archivo CSV'})

    file = request.files['csvFile']
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'El archivo no es un CSV válido'})

    try:
        csv_reader = csv.reader(file.stream.read().decode('utf-8').splitlines())
        for row in csv_reader:
            if len(row) != 6:
                print(f"Fila ignorada (longitud incorrecta): {row}")
                continue

            try:
                numbers = [int(num) for num in row[:-1]]
                special = int(row[-1])
                if len(numbers) == 5 and all(1 <= num <= 43 for num in numbers) and 1 <= special <= 16:
                    combination = Combination(numbers=numbers, special=special)
                    db.session.add(combination)
            except ValueError:
                print(f"Error al convertir: {row}")

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Endpoint para obtener todas las combinaciones
@app.route('/get_combinations', methods=['GET'])
def get_combinations():
    all_combinations = Combination.query.all()
    # result = [{'numbers': comb.numbers.split(','), 'special': comb.special} for comb in all_combinations]
    result = [{'numbers': list(map(int, comb.numbers.split(','))), 'special': comb.special} for comb in all_combinations]
    return jsonify(result)

# Función para contar frecuencias de números
def count_frequencies(combinations):
    number_frequencies = {i: 0 for i in range(1, 44)}
    special_frequencies = {i: 0 for i in range(1, 17)}

    for combination in combinations:
        for num in combination['numbers']:
            number_frequencies[num] += 1
        special_frequencies[combination['special']] += 1
    
    return number_frequencies, special_frequencies

# Función para calcular probabilidades condicionales
def calculate_conditional_probabilities(combinations):
    co_occurrences = defaultdict(lambda: defaultdict(int))

    for combination in combinations:
        numbers = combination['numbers']
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                co_occurrences[numbers[i]][numbers[j]] += 1
                co_occurrences[numbers[j]][numbers[i]] += 1

    return co_occurrences

# Endpoint para obtener probabilidades condicionales
@app.route('/conditional_probabilities', methods=['GET'])
def get_conditional_probabilities():
    combinations = [{'numbers': comb.numbers.split(','), 'special': comb.special} for comb in Combination.query.all()]
    co_occurrences = calculate_conditional_probabilities(combinations)
    return jsonify(co_occurrences)

# Función para predecir la próxima combinación
def predict_next_combination(combinations):
    for combination in combinations:
        combination['numbers'] = list(map(int, combination['numbers']))  # Convertir a enteros

    number_frequencies, special_frequencies = count_frequencies(combinations)
    next_numbers = sorted(number_frequencies, key=number_frequencies.get, reverse=True)[:5]
    next_special = max(special_frequencies, key=special_frequencies.get)
    return next_numbers, next_special

# Endpoint para predecir la próxima combinación
@app.route('/predict', methods=['GET'])
def predict():
    combinations = [{'numbers': comb.numbers.split(','), 'special': comb.special} for comb in Combination.query.all()]
    next_numbers, next_special = predict_next_combination(combinations)
    prediction = {'numbers': next_numbers, 'special': next_special}
    add_prediction_to_history(prediction)
    print(f"Predicción realizada: {prediction}")
    return jsonify(prediction)

# Función para agregar una predicción al historial
def add_prediction_to_history(prediction):
    prediction_entry = PredictionHistory(numbers=prediction['numbers'], special=prediction['special'])
    db.session.add(prediction_entry)
    db.session.commit()

# Endpoint para obtener el historial de predicciones
@app.route('/prediction_history', methods=['GET'])
def get_prediction_history():
    all_predictions = PredictionHistory.query.all()
    result = [{'numbers': pred.numbers.split(','), 'special': pred.special} for pred in all_predictions]
    return jsonify(result)

# Función para simulación Monte Carlo
def monte_carlo_simulation(combinations, num_simulations=1000):
    number_frequencies, special_frequencies = count_frequencies(combinations)

    total_numbers = sum(number_frequencies.values())
    total_special = sum(special_frequencies.values())

    number_probabilities = {num: freq / total_numbers for num, freq in number_frequencies.items()}
    special_probabilities = {num: freq / total_special for num, freq in special_frequencies.items()}

    simulations = []
    for _ in range(num_simulations):
        simulated_numbers = random.choices(
            list(number_probabilities.keys()), 
            weights=number_probabilities.values(), 
            k=5
        )
        simulated_special = random.choices(
            list(special_probabilities.keys()), 
            weights=special_probabilities.values(), 
            k=1
        )[0]

        simulations.append({'numbers': sorted(simulated_numbers), 'special': simulated_special})

    return simulations

# Endpoint para ejecutar simulación Monte Carlo
@app.route('/monte_carlo', methods=['GET'])
def run_monte_carlo():
    num_simulations = int(request.args.get('simulations', 1000))
    combinations = [{'numbers': comb.numbers.split(','), 'special': comb.special} for comb in Combination.query.all()]
    simulations = monte_carlo_simulation(combinations, num_simulations)
    return jsonify(simulations)

if __name__ == '__main__':
    app.run(debug=True)
