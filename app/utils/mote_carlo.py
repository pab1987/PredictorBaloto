import random

def monte_carlo_simulation():
    # Simula combinaciones de números aleatorios
    prediction = {"numbers": [random.randint(1, 49) for _ in range(6)], "special": random.randint(1, 10)}
    return prediction
