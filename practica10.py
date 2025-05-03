import numpy as np

def heaviside(x):
    return 1 if x >= 0 else 0

def entrenar_perceptron(tabla_verdad, nombre_compuerta, epochs=10, eta=1.0):
    # Inicializar pesos y bias
    w = np.zeros(2)
    b = 0.0

    print(f"\nEntrenando compuerta {nombre_compuerta}...\n")

    for epoch in range(epochs):
        print(f"--- Época {epoch + 1} ---")
        for x, y_esperado in tabla_verdad:
            x = np.array(x)
            y_salida = heaviside(np.dot(w, x) + b)
            error = y_esperado - y_salida

            # Actualizar pesos y bias
            w = w + eta * error * x
            b = b + eta * error

            # Mostrar resultados
            print(f"Entrada: {x}, Salida: {y_salida}, Esperado: {y_esperado}, Error: {error}")
            print(f"Pesos actualizados: {w}, Bias: {b}\n")
    
    print(f"Pesos finales para {nombre_compuerta}: {w}, Bias final: {b}")
    print("\n--- Prueba Final ---")
    for x, y_esperado in tabla_verdad:
        y_salida = heaviside(np.dot(w, x) + b)
        print(f"Entrada: {x}, Salida: {y_salida}, Esperado: {y_esperado}")

# Tablas de verdad
tabla_AND = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

tabla_OR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)
]

# Entrenar y mostrar resultados
entrenar_perceptron(tabla_AND, "AND")
entrenar_perceptron(tabla_OR, "OR")
