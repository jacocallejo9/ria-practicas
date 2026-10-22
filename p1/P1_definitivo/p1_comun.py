#!/usr/bin/env python3
"""
Práctica 1 — Utilidades comunes, sin ROS 2.

Aquí está todo lo que no necesita hablar con el robot, de modo que se
puede importar (y probar) también fuera del contenedor:

  - lectura y escritura de p1_config.json,
  - geometría del simulador: rumbos, posición de la pelota, separación,
    desviación y línea central del carril,
  - modelo del sensor: distancia a partir del tamaño del blob y ángulo a
    partir de su x, con sus ajustes por mínimos cuadrados,
  - las condiciones de fallo del protocolo (anexo B.3), que comparten el
    entorno y el evaluador.

Convenciones. El módulo sim trabaja en milímetros y grados, con el suelo
en el plano (x, z) y la orientación en la componente y de la rotación.
Aquí todo se pasa a metros y radianes. Los ángulos en el plano se miden
como atan2(dx, dz): 0 hacia +z y positivos hacia +x, que es el sentido de
la rotación y de Unity vista desde arriba.
"""

import json
import math

import numpy as np


# =====================================================================
# Configuración
# =====================================================================

def cargar_config(ruta='p1_config.json'):
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)


def guardar_config(cfg, ruta='p1_config.json'):
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write('\n')


# =====================================================================
# Ángulos y rumbos
# =====================================================================

def envolver(a):
    """Ángulo en (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def vector_rumbo(rumbo):
    """Vector unitario (x, z) hacia el que apunta un rumbo."""
    return np.array([math.sin(rumbo), math.cos(rumbo)])


def angulo_hacia(origen, destino):
    """Rumbo de la recta que va de origen a destino."""
    d = np.asarray(destino, dtype=float) - np.asarray(origen, dtype=float)
    return math.atan2(d[0], d[1])


class ConvencionRumbo:
    """Pasa la rotación y del simulador (grados) a rumbo en el plano (rad).

        rumbo = signo · rotation.y + offset

    En Unity lo esperable es signo = +1 y offset = 0, pero depende de cómo
    esté orientado el modelo del robot. Por eso no se supone: se mide con
    estimar(), que compara la rotación del guía con la dirección en la que
    avanza (el guía siempre avanza, así que su movimiento da su rumbo).
    """

    def __init__(self, signo=1, offset_deg=0.0):
        self.signo = 1 if signo >= 0 else -1
        self.offset = math.radians(offset_deg)

    def __call__(self, yaw_deg):
        return envolver(self.signo * math.radians(yaw_deg) + self.offset)

    @staticmethod
    def estimar(yaws_deg, rumbos_mov):
        """Elige signo y desfase a partir de pares (rotación, rumbo de avance).

        Para cada signo, el desfase es la media circular de
        (rumbo de avance − signo·rotación) y la longitud del vector medio R
        mide lo constante que es: R ≈ 1 con la convención correcta en
        cuanto el guía ha tomado alguna curva. Devuelve (signo, offset_deg, R).
        """
        yaw = np.radians(np.asarray(yaws_deg, dtype=float))
        mov = np.asarray(rumbos_mov, dtype=float)
        mejor = None
        for signo in (1, -1):
            dif = mov - signo * yaw
            c, s = np.mean(np.cos(dif)), np.mean(np.sin(dif))
            R = math.hypot(c, s)
            if mejor is None or R > mejor[2] + 1e-6:
                mejor = (signo, math.degrees(math.atan2(s, c)), R)
        return mejor


def posicion_pelota(pos_guia, rumbo_guia, offset_m):
    """La pelota va offset_m por detrás del centro del guía (anexo B.1)."""
    return np.asarray(pos_guia, dtype=float) - offset_m * vector_rumbo(rumbo_guia)


def desviacion(pos_seguidor, rumbo_seguidor, pos_pelota):
    """Ángulo entre el rumbo del seguidor y la dirección a la pelota (rad)."""
    return envolver(angulo_hacia(pos_seguidor, pos_pelota) - rumbo_seguidor)


# =====================================================================
# Línea central del carril (sólo verdad del terreno)
# =====================================================================

class LineaCentral:
    """Polilínea cerrada que recorre el guía, en metros.

    Se graba con  python3 p1_seguidor.py --grabar-linea . Permite calcular
    la abscisa curvilínea s de un punto (su "kilómetro" en el anillo), el
    error lateral e (distancia con signo a la línea: + hacia el interior
    del anillo) y la separación a lo largo de la línea.
    """

    def __init__(self, puntos):
        self.p = np.asarray(puntos, dtype=float)
        seg = np.roll(self.p, -1, axis=0) - self.p          # segmento i -> i+1
        self.long_seg = np.linalg.norm(seg, axis=1)
        self.tang = seg / np.maximum(self.long_seg[:, None], 1e-9)
        self.s = np.r_[0.0, np.cumsum(self.long_seg)[:-1]]
        self.L = float(self.long_seg.sum())
        # Sentido de recorrido (área con signo, x horizontal y z vertical
        # vistos desde arriba): > 0 antihorario, el interior a la izquierda.
        x, z = self.p[:, 0], self.p[:, 1]
        area = 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)
        self.signo_interior = 1.0 if area > 0 else -1.0

    def proyectar(self, q):
        """Devuelve (s, e): abscisa en m y error lateral con signo en m."""
        q = np.asarray(q, dtype=float)
        t = np.clip(np.sum((q - self.p) * self.tang, axis=1), 0.0, self.long_seg)
        pie = self.p + self.tang * t[:, None]
        dist = np.linalg.norm(q - pie, axis=1)
        i = int(np.argmin(dist))
        v = q - pie[i]
        izquierda = self.tang[i, 0] * v[1] - self.tang[i, 1] * v[0]
        e = math.copysign(dist[i], izquierda) * self.signo_interior
        return float(self.s[i] + t[i]), float(e)

    def punto(self, s):
        """Punto de la línea en la abscisa s y tangente (sentido de avance)."""
        s = s % self.L
        i = int(np.searchsorted(self.s, s, side='right') - 1)
        return self.p[i] + self.tang[i] * (s - self.s[i]), self.tang[i]

    def separacion(self, s_delante, s_detras):
        """Distancia a lo largo de la línea; negativa si 'detrás' va delante."""
        ds = (s_delante - s_detras) % self.L
        return ds - self.L if ds > self.L / 2 else ds


# =====================================================================
# Modelo del sensor (cámara + detector de blobs)
# =====================================================================

class ModeloCamara:
    """Pasa las lecturas del blob a magnitudes físicas.

    Distancia. En una cámara estenopeica el diámetro aparente de la pelota
    es proporcional a 1/d, así que su área (size) lo es a 1/d², es decir,
    d = k·size^(-1/2). El detector de RoboboSim, además, pierde píxeles del
    borde de la pelota cuando está lejos (el área cae más deprisa), así que
    se ajusta el exponente:
        d = k · size^(-p)        (estenopeica pura: p = 0,5)
    Medido en el simulador (caracterización estática): p ≈ 0,36, con un
    error del 7 % entre 0,22 y 1 m.

    Ángulo. Un punto que forma un ángulo θ con el eje óptico cae en
        x_norm = tan(θ) / tan(fov/2),    x_norm en [-1, 1]

    Dar a la política la distancia en metros, y no el tamaño en píxeles,
    es una de las medidas para la transferencia: en el robot real sólo hay
    que volver a ajustar k, b y fov en p1_config.json, y la observación
    sigue significando lo mismo.
    """

    def __init__(self, cfg):
        c = cfg['camara']
        self.k = float(c['distancia_k'])
        self.p = float(c.get('distancia_p', 0.5))
        self.fov = math.radians(c['fov_horizontal_deg'])
        self.tan_semi = math.tan(self.fov / 2.0)
        self.xc, self.xs = c['x_centro'], c['x_semiancho']
        self.yc, self.ys = c['y_centro'], c['y_semiancho']

    def x_normalizada(self, x):
        return float(np.clip((x - self.xc) / self.xs, -1.0, 1.0))

    def y_normalizada(self, y):
        return float(np.clip((y - self.yc) / self.ys, -1.0, 1.0))

    def distancia(self, size):
        return self.k * max(size, 1e-6) ** (-self.p)

    def angulo(self, x_norm):
        return math.atan(x_norm * self.tan_semi)


def ajustar_distancia(sizes, distancias):
    """Ajuste de d = k·size^(-p): recta en escala logarítmica.

    log d = log k - p·log size, por mínimos cuadrados; así el error que se
    minimiza es relativo, que es lo que importa (1 cm a 0,2 m pesa lo
    mismo que 4,5 cm a 0,9 m). Devuelve (k, p, rmse en metros).
    """
    s = np.asarray(sizes, dtype=float)
    d = np.asarray(distancias, dtype=float)
    A = np.c_[np.ones_like(s), -np.log(s)]
    (log_k, p), *_ = np.linalg.lstsq(A, np.log(d), rcond=None)
    k = math.exp(log_k)
    rmse = float(np.sqrt(np.mean((k * s ** -p - d) ** 2)))
    return float(k), float(p), rmse


def ajustar_fov(x_norm, angulos):
    """Mínimos cuadrados de tan(θ) = T · x_norm. Devuelve (fov_deg, signo, rmse_deg).

    El signo dice si una x positiva corresponde a un ángulo positivo o
    negativo en la convención de rumbos; el campo de visión es 2·atan(|T|).
    """
    x = np.asarray(x_norm, dtype=float)
    t = np.tan(np.asarray(angulos, dtype=float))
    T = float(np.sum(x * t) / np.sum(x * x))
    pred = np.arctan(T * x)
    rmse = float(np.degrees(np.sqrt(np.mean((pred - np.arctan(t)) ** 2))))
    return math.degrees(2.0 * math.atan(abs(T))), (1 if T >= 0 else -1), rmse


def proximidad_ir(valor, saturacion):
    """Lectura cruda del IR -> proximidad en [0, 1], en escala logarítmica."""
    return min(math.log1p(max(valor, 0)) / math.log1p(saturacion), 1.0)


# =====================================================================
# Protocolo de evaluación (anexo B.3)
# =====================================================================

FALLOS_PROTOCOLO = ('colision', 'guia_perdido', 'sin_pelota')


def fallo_protocolo(separacion, t_sin_pelota, p):
    """Causa de fallo del protocolo en este paso, o '' si no hay fallo.

      colision      separación < 0,12 m
      guia_perdido  separación > 0,90 m
      sin_pelota    más de 3 s seguidos sin detectar la pelota
                    (el anexo lo cuenta también como guía perdido; se
                    registra aparte para poder distinguirlo)
    """
    if separacion < p['colision_m']:
        return 'colision'
    if separacion > p['perdido_m']:
        return 'guia_perdido'
    if t_sin_pelota > p['sin_pelota_s']:
        return 'sin_pelota'
    return ''
