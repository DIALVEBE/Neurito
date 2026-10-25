import numpy as np
import pygame

# ===================================
# 1. Preparar datos de entrenamiento
# ===================================

np.random.seed(42)

# Generamos 3000 situaciones posibles.
# Cada fila representa:
# [hambre, energia, aburrimiento]
X = np.random.rand(3000, 3)

# Creamos ejemplos de decisiones correctas.
# El índice de la necesidad dominante determina la acción:
# 0 = comer, 1 = dormir, 2 = jugar.
necesidades = np.column_stack(
    (
        X[:, 0],  # Hambre
        1 - X[:, 1],  # Cansancio (energía baja)
        X[:, 2],  # Aburrimiento
    )
)

y = np.argmax(necesidades, axis=1)


# ===================================
# 2. Crear la red neuronal
# ===================================

# Tres entradas, seis neuronas ocultas,
# tres neuronas de salida.
entradas = 3
ocultas = 6
salidas = 3

# Los pesos comienzan con valores aleatorios.
W1 = np.random.randn(entradas, ocultas) * 0.5
b1 = np.zeros((1, ocultas))

W2 = np.random.randn(ocultas, salidas) * 0.5
b2 = np.zeros((1, salidas))


# ===================================
# 3. Funciones de activación
# ===================================


def relu(x):
    # Los valores negativos se convierten en cero.
    return np.maximum(0, x)


def softmax(x):
    # Convierte los resultados en probabilidades.
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# ===================================
# 4. Propagación hacia adelante
# ===================================


def predecir(X):
    # Primera capa: entradas a neuronas ocultas.
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # Segunda capa: neuronas ocultas a salidas.
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    return a2


# ===================================
# 5. Entrenar la red neuronal
# ===================================

tasa_aprendizaje = 0.1
epocas = 1200

for epoca in range(epocas):
    # PROPAGACIÓN HACIA ADELANTE
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    # CÁLCULO DEL ERROR
    # Cross entropy para clasificación.
    n = len(X)

    perdida = -np.mean(np.log(a2[np.arange(n), y] + 1e-10))

    # RETROPROPAGACIÓN
    # Gradiente de Softmax + Cross Entropy.
    dz2 = a2.copy()
    dz2[np.arange(n), y] -= 1
    dz2 /= n

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Gradiente hacia la capa oculta.
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ACTUALIZAR PESOS Y SESGOS
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1

    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2

    if epoca % 200 == 0:
        print(f"Época {epoca} | Error: {perdida:.4f}")

# Evaluamos con 1000 situaciones nuevas.
prueba = np.random.rand(1000, 3)

necesidades_prueba = np.column_stack((prueba[:, 0], 1 - prueba[:, 1], prueba[:, 2]))

respuestas = np.argmax(necesidades_prueba, axis=1)
predicciones = np.argmax(predecir(prueba), axis=1)

precision = np.mean(predicciones == respuestas)

print(f"Precisión en situaciones nuevas: {precision:.2%}")


# ===================================
# 6. Crear la mascota virtual
# ===================================

pygame.init()

ANCHO = 760
ALTO = 570

pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Neurito - Mascota con red neuronal")

reloj = pygame.time.Clock()

fuente = pygame.font.SysFont(None, 30)
fuente_pequena = pygame.font.SysFont(None, 23)

# Estado inicial de Neurito.
hambre = 0.7
energia = 0.4
aburrimiento = 0.6

acciones = ["COMER", "DORMIR", "JUGAR"]


# ===================================
# 7. Dibujar indicadores
# ===================================


def dibujar_barra(nombre, valor, y, color):
    # Nombre y porcentaje de la necesidad.
    texto = fuente_pequena.render(f"{nombre}: {valor * 100:.0f}%", True, (35, 40, 55))
    pantalla.blit(texto, (55, y - 28))

    # Fondo de la barra.
    pygame.draw.rect(pantalla, (215, 220, 230), (55, y, 260, 22), border_radius=8)

    # Relleno de acuerdo con el valor.
    pygame.draw.rect(pantalla, color, (55, y, int(260 * valor), 22), border_radius=8)


# ===================================
# 8. Dibujar a Neurito
# ===================================


def dibujar_mascota(accion):

    # Cuerpo de la mascota.
    pygame.draw.circle(pantalla, (88, 160, 235), (545, 215), 95)

    # Ojos.
    pygame.draw.circle(pantalla, (255, 255, 255), (515, 195), 18)
    pygame.draw.circle(pantalla, (255, 255, 255), (575, 195), 18)

    pygame.draw.circle(pantalla, (25, 40, 65), (519, 198), 8)
    pygame.draw.circle(pantalla, (25, 40, 65), (579, 198), 8)

    # Expresión distinta según la decisión.
    if accion == 0:
        # Boca abierta cuando quiere comer.
        pygame.draw.circle(pantalla, (40, 50, 70), (545, 250), 15)

    elif accion == 1:
        # Línea de sueño.
        pygame.draw.arc(pantalla, (40, 50, 70), (520, 235, 50, 25), 0, 3.14, 4)

    else:
        # Sonrisa cuando quiere jugar.
        pygame.draw.arc(pantalla, (40, 50, 70), (515, 230, 60, 38), 3.14, 6.28, 4)


# ===================================
# 9. Juego principal
# ===================================

ejecutando = True

while ejecutando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False

        if evento.type == pygame.KEYDOWN:
            # Cambiar las necesidades manualmente.
            if evento.key == pygame.K_h:
                hambre = min(1, hambre + 0.1)

            if evento.key == pygame.K_j:
                hambre = max(0, hambre - 0.1)

            if evento.key == pygame.K_e:
                energia = min(1, energia + 0.1)

            if evento.key == pygame.K_d:
                energia = max(0, energia - 0.1)

            if evento.key == pygame.K_a:
                aburrimiento = min(1, aburrimiento + 0.1)

            if evento.key == pygame.K_s:
                aburrimiento = max(0, aburrimiento - 0.1)

            # Nueva situación aleatoria.
            if evento.key == pygame.K_r:
                hambre, energia, aburrimiento = np.random.rand(3)

    # La red neuronal analiza el estado actual.
    estado = np.array([[hambre, energia, aburrimiento]])

    probabilidades = predecir(estado)[0]

    # Selecciona la acción con mayor probabilidad.
    decision = int(np.argmax(probabilidades))

    # Dibujamos la interfaz.
    pantalla.fill((245, 247, 252))

    titulo = fuente.render("NEURITO - MI PRIMER CEREBRO ARTIFICIAL", True, (35, 45, 65))
    pantalla.blit(titulo, (55, 30))

    dibujar_barra("Hambre", hambre, 120, (240, 150, 75))
    dibujar_barra("Energia", energia, 200, (65, 180, 115))
    dibujar_barra("Aburrimiento", aburrimiento, 280, (110, 135, 240))

    dibujar_mascota(decision)

    texto_accion = fuente.render(f"Quiero: {acciones[decision]}", True, (35, 45, 65))
    pantalla.blit(texto_accion, (430, 340))

    # Mostrar las probabilidades de cada salida.
    for i, nombre in enumerate(acciones):
        texto = fuente_pequena.render(
            f"{nombre}: {probabilidades[i] * 100:.1f}%", True, (40, 45, 60)
        )
        pantalla.blit(texto, (430, 390 + i * 29))

    ayuda1 = fuente_pequena.render(
        "H/J: hambre +/-    E/D: energia +/-", True, (65, 70, 90)
    )
    ayuda2 = fuente_pequena.render(
        "A/S: aburrimiento +/-    R: estado aleatorio", True, (65, 70, 90)
    )

    pantalla.blit(ayuda1, (55, 490))
    pantalla.blit(ayuda2, (55, 520))

    pygame.display.flip()
    reloj.tick(30)

pygame.quit()
