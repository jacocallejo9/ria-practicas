# Archivo de configuración global para la Práctica 1 (RIA).


# SELECCIÓN DE ENTORNO
# Cambiar a True cuando se ejecute en el robot real del laboratorio
ES_ENTORNO_REAL = False


# ESPACIO DE ACCIÓN Y TIEMPOS DE CONTROL
# Frecuencia/Periodo de control
PASO_S = 0.2  # Duración de cada paso del entorno (en segundos)

# Límites físicos de velocidad para el robot seguidor (unidades SI)
# v: velocidad lineal (m/s)
# w: velocidad angular (rad/s)
V_MIN = -0.05  # Retroceso suave para evitar colisión si el guía frena brusco
V_MAX = 0.25   # Velocidad máxima lineal hacia delante
W_MAX = 1.5    # Velocidad angular máxima en rad/s


# PARAMETROS DE CALIBRACIÓN DE CÁMARA / BLOB VERDE
# Valores ajustados mediante caracterización previa (paso 3)
if not ES_ENTORNO_REAL:
    # --- PARÁMETROS SIMULADOR ---
    BLOB_SIZE_MAX = 400.0   # Tamaño aproximado del blob verde cuando está a colisión (< 0.12m)
    BLOB_SIZE_MIN = 10.0    # Tamaño aproximado a la distancia límite de visión (~ 0.90m)
    CAM_ANCHO_PX = 640.0    # Ancho de la resolución de la cámara en píxeles
else:
    # --- PARÁMETROS ENTORNO REAL ---
    # Ajustar estos valores tras ejecutar la caracterización en el robot real
    BLOB_SIZE_MAX = 450.0
    BLOB_SIZE_MIN = 12.0
    CAM_ANCHO_PX = 640.0


# PROTOCOLO DE EVALUACIÓN Y LÍMITES FÍSICOS (ANEXO B)
# Distancias en metros
SEPARACION_OBJETIVO_MIN = 0.20  # Límite inferior de la banda objetivo
SEPARACION_OBJETIVO_MAX = 0.45  # Límite superior de la banda objetivo
SEPARACION_CENTRO_BANDA = (SEPARACION_OBJETIVO_MIN + SEPARACION_OBJETIVO_MAX) / 2.0  # 0.325 m

DISTANCIA_COLISION = 0.12        # Colisión si distancia < 0.12 m
DISTANCIA_PERDIDA = 0.90         # Guía perdido si distancia > 0.90 m
TIEMPO_PERDIDO_S = 3.0           # Guía perdido si no se ve durante más de 3.0 s seguidos

DESVIACION_ANGULO_MAX_OK = 15.0  # Ángulo de desviación considerado "bueno" para métricas (grados)


# FUNCION DE RECOMPENSA (PESOS Y COEFICIENTES)
RECOMPENSA = {
    'k_separacion': 1.0,         # Peso para el término gaussiano de distancia objetivo
    'k_desviacion': 0.5,         # Penalización por no tener la pelota centrada en X
    'k_visibilidad': 0.8,        # Penalización por paso cuando no se detecta el blob
    'k_suavidad': 0.1,           # Penalización por cambios bruscos en la velocidad angular (dw)
    'penalizacion_fallo': 10.0,  # Penalización terminal por fallo (colisión o perder al guía)
}


# CONFIGURACIONES DEL VEHÍCULO GUÍA PARA EVALUACIÓN (ANEXO B.2)
CONFIGURACIONES_GUIA = {
    'C1': {
        'nombre': 'Lenta constante',
        'modo': 'crucero',
        'v_min': 10,
        'v_max': 10
    },
    'C2': {
        'nombre': 'Contraste',
        'modo': 'crucero',
        'v_min': 10,
        'v_max': 40
    },
    'C3': {
        'nombre': 'Fluctuante',
        'modo': 'variable',
        'v_min': 10,
        'v_max': 40
    }
}