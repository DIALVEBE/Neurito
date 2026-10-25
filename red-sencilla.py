# Entradas de nuestra mascota
hambre = 0.8
energia = 0.2
aburrimiento = 0.5

# Pesos iniciales de la neurona
w1 = 0.7
w2 = -0.5
w3 = 0.3

# Sesgo
b = 0.1

# Suma ponderada
z = hambre * w1 + energia * w2 + aburrimiento * w3 + b

# Función de activación ReLU
salida = max(0, z)

print("Resultado de la neurona:", salida)
