from pathlib import Path

import numpy as np
import pygame


# ===================================
# 1. Preparar datos de entrenamiento
# ===================================

np.random.seed(42)

# Generamos 3000 situaciones posibles.
# Cada fila contiene hambre, energia y aburrimiento.
X = np.random.rand(3000, 3)

# Creamos las decisiones correctas.
# 0 = comer, 1 = dormir, 2 = jugar.
necesidades = np.column_stack((X[:, 0], 1 - X[:, 1], X[:, 2]))

y = np.argmax(necesidades, axis=1)


# ===================================
# 2. Crear la red neuronal
# ===================================

# Arquitectura: 3 entradas, 6 ocultas, 3 salidas.
entradas = 3
ocultas = 6
salidas = 3

# Pesos y sesgos iniciales.
W1 = np.random.randn(entradas, ocultas) * 0.5
b1 = np.zeros((1, ocultas))

W2 = np.random.randn(ocultas, salidas) * 0.5
b2 = np.zeros((1, salidas))


# ===================================
# 3. Funciones de activación
# ===================================


def relu(x):
    # Convierte los valores negativos en cero.
    return np.maximum(0, x)


def softmax(x):
    # Transforma las salidas en probabilidades.
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# ===================================
# 4. Propagación hacia adelante
# ===================================


def predecir(datos):
    # Primera capa.
    z1 = datos @ W1 + b1
    a1 = relu(z1)

    # Segunda capa.
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    return a2


# ===================================
# 5. Entrenar la red neuronal
# ===================================

tasa_aprendizaje = 0.1
epocas = 1200

for epoca in range(epocas):
    # Propagación hacia adelante.
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    # Calcular la pérdida.
    n = len(X)

    perdida = -np.mean(np.log(a2[np.arange(n), y] + 1e-10))

    # Retropropagación: capa de salida.
    dz2 = a2.copy()
    dz2[np.arange(n), y] -= 1
    dz2 /= n

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Retropropagación: capa oculta.
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Actualizar pesos y sesgos.
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1

    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2

    if epoca % 200 == 0:
        print(f"Época {epoca} | Error: {perdida:.4f}")


# ===================================
# 6. Evaluar la red neuronal
# ===================================

prueba = np.random.rand(1000, 3)

necesidades_prueba = np.column_stack((prueba[:, 0], 1 - prueba[:, 1], prueba[:, 2]))

respuestas = np.argmax(necesidades_prueba, axis=1)
predicciones = np.argmax(predecir(prueba), axis=1)

precision = np.mean(predicciones == respuestas)

print(f"Precisión en situaciones nuevas: {precision:.2%}")


# ===================================
# 7. Configurar Pygame
# ===================================

pygame.init()

ANCHO = 900
ALTO = 600

pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Neurito - Mascota con cerebro artificial")

reloj = pygame.time.Clock()

fuente = pygame.font.SysFont(None, 34)
fuente_pequena = pygame.font.SysFont(None, 24)
fuente_titulo = pygame.font.SysFont(None, 40)

# Carpeta que contiene nuestros sprites.
RUTA_ASSETS = Path(__file__).resolve().parent / "assets"

# Tamaño de las imágenes en pantalla.
TAMANO_MASCOTA = (350, 350)


# ===================================
# 8. Cargar imágenes de la mascota
# ===================================


def cargar_imagen(nombre):
    # Buscar el archivo en la carpeta assets.
    ruta = RUTA_ASSETS / nombre

    # Cargar PNG conservando su transparencia.
    imagen = pygame.image.load(str(ruta)).convert_alpha()

    # Ajustar el tamaño de la imagen.
    imagen = pygame.transform.smoothscale(imagen, TAMANO_MASCOTA)

    return imagen


# Cada estado contiene los frames de su animación.
animaciones = {
    "neutral": [
        cargar_imagen("neutral_1.png"),
        cargar_imagen("neutral_2.png"),
        cargar_imagen("neutral_3.png"),
    ],
    "comer": [cargar_imagen("comer_1.png"), cargar_imagen("comer_2.png")],
    "dormir": [cargar_imagen("dormir_1.png"), cargar_imagen("dormir_2.png")],
    "jugar": [cargar_imagen("jugar_1.png"), cargar_imagen("jugar_2.png")],
}


# ===================================
# 9. Configurar las animaciones
# ===================================

# Tiempo que permanece cada imagen (milisegundos).
VELOCIDAD_ANIMACION = 500

# Estado visual inicial.
estado_actual = "neutral"

# Índice de la imagen que estamos mostrando.
frame_actual = 0

# Momento del último cambio de imagen.
ultimo_cambio = pygame.time.get_ticks()


def actualizar_animacion(nuevo_estado):
    global estado_actual
    global frame_actual
    global ultimo_cambio

    # Tiempo actual de Pygame en milisegundos.
    ahora = pygame.time.get_ticks()

    # Si cambia el estado, reiniciamos la animación.
    if nuevo_estado != estado_actual:
        estado_actual = nuevo_estado
        frame_actual = 0
        ultimo_cambio = ahora

    # Si transcurrieron 500 ms, avanzamos de imagen.
    elif ahora - ultimo_cambio >= VELOCIDAD_ANIMACION:
        # Pasar a la siguiente imagen.
        # El módulo (%) permite regresar al principio.
        frame_actual = (frame_actual + 1) % len(animaciones[estado_actual])

        # Guardar el momento del cambio.
        ultimo_cambio = ahora

    # Devolver la imagen correspondiente.
    return animaciones[estado_actual][frame_actual]


# ===================================
# 10. Estado inicial de la mascota
# ===================================

hambre = 0.2
energia = 0.9
aburrimiento = 0.2

acciones = ["comer", "dormir", "jugar"]


# ===================================
# 11. Dibujar indicadores
# ===================================


def dibujar_barra(nombre, valor, y, color):

    texto = fuente_pequena.render(f"{nombre}: {valor * 100:.0f}%", True, (35, 40, 55))
    pantalla.blit(texto, (55, y - 30))

    # Fondo de la barra.
    pygame.draw.rect(pantalla, (215, 220, 230), (55, y, 300, 24), border_radius=8)

    # Barra de progreso.
    pygame.draw.rect(pantalla, color, (55, y, int(300 * valor), 24), border_radius=8)


# ===================================
# 12. Determinar el estado visual
# ===================================


def obtener_estado(hambre, energia, aburrimiento, decision):

    # Si sus necesidades están satisfechas,
    # mostramos el estado neutro.
    #
    # Esta condición no pertenece a la red neuronal.
    # Es una regla adicional para la interfaz.
    if hambre < 0.4 and energia > 0.6 and aburrimiento < 0.4:
        return "neutral"

    # En los demás casos utilizamos
    # la decisión de la red neuronal.
    return acciones[decision]


# ===================================
# 13. Juego principal
# ===================================

ejecutando = True

while ejecutando:
    # -----------------------------------
    # Procesar eventos del teclado
    # -----------------------------------

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False

        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_ESCAPE:
                ejecutando = False

            # Modificar el hambre.
            if evento.key == pygame.K_h:
                hambre = min(1, hambre + 0.1)

            if evento.key == pygame.K_j:
                hambre = max(0, hambre - 0.1)

            # Modificar la energía.
            if evento.key == pygame.K_e:
                energia = min(1, energia + 0.1)

            if evento.key == pygame.K_d:
                energia = max(0, energia - 0.1)

            # Modificar el aburrimiento.
            if evento.key == pygame.K_a:
                aburrimiento = min(1, aburrimiento + 0.1)

            if evento.key == pygame.K_s:
                aburrimiento = max(0, aburrimiento - 0.1)

            # Generar un estado aleatorio.
            if evento.key == pygame.K_r:
                hambre, energia, aburrimiento = np.random.rand(3)

            # Recuperar el estado tranquilo.
            if evento.key == pygame.K_n:
                hambre = 0.2
                energia = 0.9
                aburrimiento = 0.2

    # -----------------------------------
    # Consultar la red neuronal
    # -----------------------------------

    estado = np.array([[hambre, energia, aburrimiento]])

    probabilidades = predecir(estado)[0]

    # Elegir la salida con mayor probabilidad.
    decision = int(np.argmax(probabilidades))

    # Determinar qué estado debemos mostrar.
    nuevo_estado = obtener_estado(hambre, energia, aburrimiento, decision)

    # Actualizar el frame de la animación.
    imagen_mascota = actualizar_animacion(nuevo_estado)

    # -----------------------------------
    # Dibujar interfaz
    # -----------------------------------

    pantalla.fill((250, 245, 252))

    titulo = fuente_titulo.render("NEURITO - MI CEREBRO ARTIFICIAL", True, (65, 50, 85))
    pantalla.blit(titulo, (55, 30))

    # Indicadores de necesidades.
    dibujar_barra("Hambre", hambre, 130, (240, 150, 75))

    dibujar_barra("Energia", energia, 225, (65, 180, 115))

    dibujar_barra("Aburrimiento", aburrimiento, 320, (110, 135, 240))

    # -----------------------------------
    # Mostrar mascota animada
    # -----------------------------------

    posicion_mascota = imagen_mascota.get_rect(center=(660, 245))

    pantalla.blit(imagen_mascota, posicion_mascota)

    # Mostrar el estado actual.
    texto_estado = fuente.render(f"Estado: {nuevo_estado.upper()}", True, (90, 65, 110))

    rect_estado = texto_estado.get_rect(center=(660, 440))

    pantalla.blit(texto_estado, rect_estado)

    # -----------------------------------
    # Mostrar predicciones de la red
    # -----------------------------------

    for i, nombre in enumerate(acciones):
        texto = fuente_pequena.render(
            f"{nombre.upper()}: {probabilidades[i] * 100:.1f}%", True, (65, 60, 80)
        )

        pantalla.blit(texto, (55, 400 + i * 30))

    # -----------------------------------
    # Mostrar controles
    # -----------------------------------

    ayuda1 = fuente_pequena.render(
        "H/J: Hambre +/-     E/D: Energia +/-", True, (90, 90, 110)
    )

    ayuda2 = fuente_pequena.render(
        "A/S: Aburrimiento +/-     R: Aleatorio", True, (90, 90, 110)
    )

    ayuda3 = fuente_pequena.render(
        "N: Estado neutro     ESC: Salir", True, (90, 90, 110)
    )

    pantalla.blit(ayuda1, (55, 500))
    pantalla.blit(ayuda2, (55, 530))
    pantalla.blit(ayuda3, (55, 560))

    # Actualizar la pantalla.
    pygame.display.flip()

    # Mantener 30 fotogramas por segundo.
    reloj.tick(30)

pygame.quit()
