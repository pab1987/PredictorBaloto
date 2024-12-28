# app/utils/prediction_utils.py

import random
from collections import defaultdict


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


# Función para contar frecuencias de números
def count_frequencies(combinations):
    number_frequencies = {i: 0 for i in range(1, 44)}
    special_frequencies = {i: 0 for i in range(1, 17)}

    for combination in combinations:
        for num in combination['numbers']:
            number_frequencies[num] += 1
        special_frequencies[combination['special']] += 1
    
    return number_frequencies, special_frequencies
