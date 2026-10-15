#!/usr/bin/env python3
"""
Práctica 1 — Función de recompensa del seguidor.

--------------------------------------------------------------------
IDEA
--------------------------------------------------------------------

Una suma de términos con pesos puestos a mano (+1 por estar en la banda,
+0,5 por el carril, +0,2 por ver la pelota...) obliga a decidir cuánto
vale cada objetivo frente a los demás, y esa decisión sólo se puede
calibrar por prueba y error. Aquí no hay pesos. Cada objetivo se expresa
como un GRADO DE CUMPLIMIENTO en [0, 1], construido sólo con umbrales que
ya fija el problema (el protocolo del anexo B y la geometría de la
cámara), y los objetivos se combinan con un Y lógico, el producto:

    r_tarea = C_sep(d) · C_ori(θ)

  - Un paso vale 1 sólo si se cumplen todos los objetivos a la vez y 0 si
    cualquiera de ellos falla del todo: ninguno compensa a otro, así que
    no hay que decidir cuánto vale la separación frente a la orientación.
  - Todo paso vivo vale entre 0 y 1. Terminar antes de tiempo cuesta, sin
    más, toda la recompensa futura que se deja de cobrar.

Sobre esa base hay dos términos más, que tampoco son pesos sino
consecuencia de la formulación:

  - Moldeado basado en potencial (Ng, Harada y Russell, 1999):
        F = γ·Φ(d') - Φ(d)
    Φ mide lo cerca que está la separación del centro de la banda. F es,
    en la práctica, la VELOCIDAD RELATIVA con signo: positivo si el
    seguidor se mueve hacia el centro de la banda, negativo si se aleja de
    él. Es lo que enseña a adaptar la velocidad cuando el guía acelera o
    frena: si el guía acelera, d crece y F < 0 hasta que el seguidor iguala
    su velocidad. Está demostrado que este término no cambia la política
    óptima (sólo acelera el aprendizaje), así que tampoco hay que
    calibrar su escala frente a r_tarea. Requisitos: el mismo γ que el
    algoritmo y Φ = 0 en los estados terminales.
  - Fallo: -1 en el paso en el que el episodio termina por un fallo. Es
    el valor opuesto al de un paso perfecto: hace que fallar sea peor que
    el peor paso vivo (0), que si no serían indistinguibles.

--------------------------------------------------------------------
TÉRMINOS, MAGNITUD DE LA QUE DEPENDEN Y CÓMO LA INFIERE EL ROBOT
--------------------------------------------------------------------

C_sep(d)   d = separación real seguidor-pelota (módulo sim).
           Inferible: el área del blob es ~1/d² (observación [3]).
           Trapecio definido por el protocolo:
               0 en d ≤ 0,12 (colisión)   ->  1 en d = 0,20
               1 en toda la banda [0,20, 0,45]
               1 en d = 0,45   ->   0 en d ≥ 0,90 (guía perdido)

C_ori(θ)   θ = ángulo de la pelota respecto al eje de la cámara,
           calculado con la x del blob (observación directa [1]).
               1 si |θ| ≤ 15° (umbral de desviación del protocolo)
               lineal hasta 0 en |θ| = fov/2 (borde de la imagen: ahí
               se pierde la pelota)
               0 si la pelota no se ve
           La tolerancia de 15° deja que en las curvas la pelota se
           desplace en la imagen sin penalización: girar de golpe hacia
           ella es lo que hace recortar la curva y salirse del carril.

Φ(d)       Triángulo: 1 en el centro de la banda (0,325 m), 0 en los
           límites de fallo (0,12 y 0,90 m). Inferible como C_sep, y su
           variación lo es por el cambio del tamaño del blob (obs. [5]).


Ejecutar este archivo imprime una tabla con los valores de cada término,
útil para la memoria:   python3 p1_recompensa.py
"""

import math


def rampa(v, cero, uno):
    """0 en 'cero', 1 en 'uno', lineal entre ambos y saturada fuera.

    Sirve en los dos sentidos: si cero > uno la rampa es decreciente.
    """
    return min(max((v - cero) / (uno - cero), 0.0), 1.0) + 0.0   # + 0.0: evita el -0.0


class Recompensa:

    def __init__(self, cfg):
        p = cfg['protocolo']
        self.d_col = p['colision_m']
        self.d_min, self.d_max = p['banda_m']
        self.d_perd = p['perdido_m']
        self.d_obj = 0.5 * (self.d_min + self.d_max)

        self.th_ok = math.radians(p['desviacion_ok_deg'])
        self.th_lim = math.radians(cfg['camara']['fov_horizontal_deg']) / 2.0

        car = cfg['carril']
        self.e_ok = (car['ancho_carril_m'] - car['ancho_robot_m']) / 2.0
        self.e_lim = car['ancho_carril_m'] / 2.0
        self.usar_carril = bool(car.get('en_recompensa', False))

        r = cfg['recompensa']
        self.gamma = r['gamma']
        self.moldeado = bool(r['moldeado_potencial'])
        self.valor_fallo = r['valor_fallo']

    # -----------------------------------------------------------------
    # Grados de cumplimiento, en [0, 1]
    # -----------------------------------------------------------------

    def cumplimiento_separacion(self, d):
        return min(rampa(d, self.d_col, self.d_min), rampa(d, self.d_perd, self.d_max))

    def cumplimiento_orientacion(self, theta, visible):
        if not visible or theta is None:
            return 0.0
        if self.th_lim <= self.th_ok:          # campo de visión más estrecho que la tolerancia
            return 1.0
        return rampa(abs(theta), self.th_lim, self.th_ok)

    def cumplimiento_carril(self, e):
        if not self.usar_carril or e is None or math.isnan(e):
            return 1.0
        return rampa(abs(e), self.e_lim, self.e_ok)

    def potencial(self, d):
        return min(rampa(d, self.d_col, self.d_obj), rampa(d, self.d_perd, self.d_obj))

    # -----------------------------------------------------------------

    def __call__(self, d_ant, d, theta, visible, e=float('nan'), fallo=False):
        """Recompensa del paso y su desglose.

        d_ant   separación del paso anterior (None en el primer paso)
        d       separación actual, m
        theta   ángulo de la pelota según la x del blob, rad (None si no se ve)
        visible si el detector ve la pelota
        e       error lateral, m (nan si no hay línea central)
        fallo   el episodio termina en este paso por un fallo
        """
        c_sep = self.cumplimiento_separacion(d)
        c_ori = self.cumplimiento_orientacion(theta, visible)
        c_car = self.cumplimiento_carril(e)
        # El paso en el que se falla no es un paso cumplido.
        tarea = 0.0 if fallo else c_sep * c_ori * c_car

        moldeado = 0.0
        if self.moldeado and d_ant is not None:
            phi_sig = 0.0 if fallo else self.potencial(d)
            moldeado = self.gamma * phi_sig - self.potencial(d_ant)

        castigo = self.valor_fallo if fallo else 0.0
        total = tarea + moldeado + castigo
        return float(total), {'c_sep': c_sep, 'c_ori': c_ori, 'c_carril': c_car,
                              'tarea': tarea, 'moldeado': moldeado, 'fallo': castigo}


# =====================================================================
# Tabla de valores para la memoria
# =====================================================================

if __name__ == '__main__':
    from p1_comun import cargar_config

    R = Recompensa(cargar_config())
    print('Separación d (m)   C_sep   Φ(d)')
    for d in [0.10, 0.12, 0.16, 0.20, 0.25, 0.325, 0.40, 0.45, 0.55, 0.70, 0.90, 1.0]:
        print('   {:5.3f}          {:4.2f}   {:4.2f}'.format(
            d, R.cumplimiento_separacion(d), R.potencial(d)))

    print('\nÁngulo |θ| (°)    C_ori   (fov/2 = {:.1f}°)'.format(math.degrees(R.th_lim)))
    for a in [0, 5, 10, 15, 20, 25, 30, 35]:
        print('   {:4.0f}            {:4.2f}'.format(
            a, R.cumplimiento_orientacion(math.radians(a), True)))

    print('\nMoldeado F = γΦ(d\') - Φ(d) en un paso de 0,3 s con d en la banda:')
    for vrel in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        d0 = 0.40
        d1 = d0 + vrel * 0.3
        print('   velocidad de alejamiento {:+.2f} m/s  ->  F = {:+.3f}'.format(
            vrel, R.gamma * R.potencial(d1) - R.potencial(d0)))
