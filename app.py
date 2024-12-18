
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import random
import csv
import random
import joblib
from collections import defaultdict
from datetime import datetime

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__)

# Ruta para el índice (usuario)
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para el panel de administrador
@app.route('/admin')
def admin():
    return render_template('admin.html')


# Usar la URL de la base de datos externa proporcionada por Render
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///default.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializar la base de datos
db = SQLAlchemy(app)

# Permitir solo el origen específico
CORS(app, resources={r"/add_combination": {"origins": "*"},
                     r"/upload_csv": {"origins": "*"}})

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
    
def get_data():
    with app.app_context():  # Agrega el contexto de aplicación aquí
        combinations = Combination.query.all()
        
        X = []
        y = []

        for comb in combinations:
            numbers = list(map(int, comb.numbers.split(',')))
            special = comb.special
            X.append(numbers)
            y.append(special)

        return pd.DataFrame(X), y

def train_and_save_model():
    X, y = get_data()
    model = RandomForestClassifier(
        n_estimators=200, # Más árboles para mayor precisión
        max_depth=10,       # Limitar la profundidad para evitar sobreajuste
        min_samples_split=5, # Mejor generalización
        random_state=42     # Reproducibilidad
        
    )
    
    # Evaluar la precisión con validación cruzada
    accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Precisión promedio (cross-validation): {accuracy_scores.mean():.4f}")
    
    # Evaluar F1-Score con validación cruzada
    f1_scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score, average='weighted'))
    print(f"F1-Score promedio (cross-validation): {f1_scores.mean():.4f}")
    
    # Evaluar Precisión con validación cruzada
    precision_scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(precision_score, average='weighted'))
    print(f"Precisión promedio (cross-validation): {precision_scores.mean():.4f}")
    
    # Entrenamiento final
    model.fit(X, y)
    
    joblib.dump(model, 'trained_model.pkl')
    print("Modelo entrenado y guardado como 'trained_model.pkl'.")

# Endpoint para agregar una combinación
@app.route('/add_combination', methods=['POST'])
def add_combination():
    data = request.get_json()

    numbers = data.get('numbers')
    special = data.get('special')

    if not numbers or len(numbers) != 5 or not special:
        return jsonify({'success': False, 'error': 'Datos inválidos'})

    """ try:
        numbers = [int(num) for num in numbers]
        special = int(special)
    except ValueError:
        return jsonify({'success': False, 'error': 'Error al convertir números'}) """
        
    try:
        numbers = [int(num) for num in numbers]
        special = int(special)

        if len(set(numbers)) != 5:  # Verificar que los números sean únicos
            return jsonify({'success': False, 'error': 'Los números no deben repetirse.'})

        if not all(1 <= num <= 43 for num in numbers):  # Rango válido
            return jsonify({'success': False, 'error': 'Los números deben estar entre 1 y 43.'})

        if not (1 <= special <= 16):  # Verifica el rango del número especial
            return jsonify({'success': False, 'error': 'El número especial debe estar entre 1 y 16.'})

    except ValueError:
        return jsonify({'success': False, 'error': 'Error al convertir números.'})

    # Guardar en la base de datos
    combination = Combination(numbers=numbers, special=special)
    db.session.add(combination)
    db.session.commit()

    print(f"Combinación añadida: {combination}")  # Mensaje de depuración
    
    # Entrenar y guardar el modelo después de añadir la nueva combinación
    train_and_save_model()

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

    
# Función mejorada para predecir la próxima combinación
def predict_next_combination(combinations):
    # Convertir a enteros
    for combination in combinations:
        combination['numbers'] = list(map(int, combination['numbers']))

    # Contar frecuencias
    number_frequencies, special_frequencies = count_frequencies(combinations)
    
    # Probabilidades condicionales
    co_occurrences = calculate_conditional_probabilities(combinations)

    # Selección ponderada de los números
    total_numbers = sum(number_frequencies.values())
    number_probabilities = {num: freq / total_numbers for num, freq in number_frequencies.items()}

    # Empezar a construir la combinación predicha
    selected_numbers = []
    while len(selected_numbers) < 5:
        # Selección ponderada del número inicial
        if not selected_numbers:
            next_num = random.choices(
                list(number_probabilities.keys()), 
                weights=number_probabilities.values(), 
                k=1
            )[0]
            selected_numbers.append(next_num)
        else:
            # Usar probabilidades condicionales basadas en co-ocurrencias
            last_selected = selected_numbers[-1]
            next_candidates = list(co_occurrences[last_selected].keys())
            if next_candidates:
                weights = [co_occurrences[last_selected][candidate] for candidate in next_candidates]
                next_num = random.choices(next_candidates, weights=weights, k=1)[0]
            else:
                # En caso de no tener co-ocurrencias, seleccionar otro número ponderado por frecuencia
                remaining_numbers = set(number_probabilities.keys()) - set(selected_numbers)
                remaining_probs = [number_probabilities[num] for num in remaining_numbers]
                next_num = random.choices(list(remaining_numbers), weights=remaining_probs, k=1)[0]

            if next_num not in selected_numbers:
                selected_numbers.append(next_num)

    # Selección ponderada para el número especial
    total_special = sum(special_frequencies.values())
    special_probabilities = {num: freq / total_special for num, freq in special_frequencies.items()}
    next_special = random.choices(
        list(special_probabilities.keys()), 
        weights=special_probabilities.values(), 
        k=1
    )[0]

    return sorted(selected_numbers), next_special

# Cargar el modelo al inicio de la aplicación
model = joblib.load('trained_model.pkl')

@app.route('/predict', methods=['GET'])
def predict():

    combinations = [{'numbers': list(map(int, comb.numbers.split(','))), 'special': comb.special} for comb in Combination.query.all()]
    next_numbers, next_special = predict_next_combination(combinations)
    
    # Aquí puedes usar el modelo para predecir el número especial
    prediction_input = [next_numbers]  # Asegúrate de que tenga la forma correcta
    predicted_special = model.predict([next_numbers])[0]  # Predicción usando el modelo

    # Convertir predicted_special a un tipo nativo
    predicted_special = int(predicted_special)  # Asegúrate de que sea un int

    prediction = {'numbers': next_numbers, 'special': predicted_special}
    add_prediction_to_history(prediction)
    
    # Obtener la hora actual
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Predicción realizada: {prediction}")
    
    # Imprimir las características importantes del modelo
    print("******** Características importantes ********** ", model.feature_importances_)
    print("Número de estimadores:", model.n_estimators)
    
    response = {
        "estado": "exitoso",  # Indicador de éxito
        "fecha_hora": timestamp,  # Fecha y hora actual
        "mensaje": "La predicción se ha generado con éxito. Aquí están los números recomendados para la próxima jugada:",
        "informacion_adicional": "Estas son las características clave que el modelo utilizó para hacer la predicción. El modelo usó 100 estimadores para generar esta predicción.",
        "prediccion": {
            "balotas": prediction["numbers"],
            "super_balota": prediction["special"]
        },
        
    }

    return jsonify(response)



def add_prediction_to_history(prediction):
    # Convierte los números y el número especial a tipos nativos de Python
    numbers = prediction['numbers']  # Asegúrate de que sea una cadena o lista
    special = int(prediction['special'])  # Convierte a int

    new_prediction = PredictionHistory(numbers=numbers, special=special)
    db.session.add(new_prediction)
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
