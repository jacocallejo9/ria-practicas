#!/usr/bin/env python3
"""
Práctica 2 — Función de calidad (fitness) del seguidor.

Implementa P2_funcion_calidad.md:

    F = 20 · n/N  +  70 · (1/N) · Σ q_t  +  10 · [completado]      F en [0, 100]

  N  pasos máximos del episodio (100 por defecto, ~60 s).
  n  pasos vividos sin fallar. Término de SUPERVIVENCIA: da gradiente desde
     la primera generación (el que pierde al guía a los 8 s puntúa más que
     el que lo pierde a los 2 s aunque los dos lo hagan mal) y no depende
     de lo bien que se hace cada paso.
  q_t calidad de cada paso, en [0, 1]. Los pasos no vividos cuentan 0: es
     la penalización implícita por fallar. El paso en que se falla no suma.
  [completado] llega a N pasos sin fallar.

Calidad de cada paso, con datos PRIVILEGIADOS del simulador (la red no los
recibe; vienen en el `info` de step()):

  C_sep(d)   d = separación real. 1 en la banda [0,20, 0,45] m; recta hasta
             0 en 0,12 m (colisión) y en 0,90 m (guía perdido).
  C_ori(θ)   θ = desviación real. 1 si |θ| <= 15°; recta hasta 0 en 23,3°
             (mitad del campo de visión: la pelota sale de la imagen).
  C_car(e)   e = error lateral respecto a la línea central. 1 si |e| <=
             0,045 m (el robot de 0,16 m cabe en el carril de 0,25 m); recta
             hasta 0 en 0,125 m (el centro del robot pisa la raya).

  A (producto)  q = C_sep · C_ori · C_car       (las tres a la vez)
  B (suma)      q = 0,5·C_sep + 0,3·C_ori + 0,2·C_car

Los umbrales salen de p1_config.json (protocolo, camara, carril), así que no
hay números mágicos aquí. Sin línea central (e = nan) C_car vale 1.

    python3 p2_calidad.py        # tabla de valores para la memoria
"""

import math

PESOS_B = (0.5, 0.3, 0.2)       # separación, orientación, carril
PESOS_F = (20.0, 70.0, 10.0)    # supervivencia, calidad por paso, completado (suman 100)


def rampa(v, cero, uno):
    """0 en 'cero', 1 en 'uno', lineal entre ambos y saturada fuera."""
    return min(max((v - cero) / (uno - cero), 0.0), 1.0) + 0.0


class Calidad:

    def __init__(self, cfg, formulacion='A', n_pasos=100, pesos=PESOS_F):
        if formulacion not in ('A', 'B'):
            raise ValueError('formulación desconocida: ' + str(formulacion))
        p = cfg['protocolo']
        self.formulacion = formulacion
        self.N = int(n_pasos)
        self.p_vida, self.p_paso, self.p_fin = (float(x) for x in pesos)
        self.d_col = p['colision_m']
        self.d_min, self.d_max = p['banda_m']
        self.d_perd = p['perdido_m']
        self.th_ok = p['desviacion_ok_deg']
        self.th_lim = cfg['camara']['fov_horizontal_deg'] / 2.0
        car = cfg['carril']
        self.e_ok = (car['ancho_carril_m'] - car['ancho_robot_m']) / 2.0     # 0,045
        self.e_lim = car['ancho_carril_m'] / 2.0                             # 0,125

    # ------------------------------------------------------------------
    def c_sep(self, d):
        return min(rampa(d, self.d_col, self.d_min), rampa(d, self.d_perd, self.d_max))

    def c_ori(self, theta_deg):
        return rampa(abs(theta_deg), self.th_lim, self.th_ok)

    def c_car(self, e):
        if e is None or math.isnan(e):
            return 1.0
        return rampa(abs(e), self.e_lim, self.e_ok)

    def combinar(self, c_sep, c_ori, c_car):
        if self.formulacion == 'A':
            return c_sep * c_ori * c_car
        a, b, c = PESOS_B
        return a * c_sep + b * c_ori + c * c_car

    def paso(self, info):
        """(q, (C_sep, C_ori, C_car)) de un paso, a partir del info de step()."""
        t = (self.c_sep(info['separacion']),
             self.c_ori(info['desviacion_deg']),
             self.c_car(info.get('error_lateral', float('nan'))))
        return self.combinar(*t), t

    def total(self, suma_q, vivos, completado):
        """Calidad del episodio a partir de la suma de q, los pasos vividos sin
        fallar y si se ha completado."""
        return (self.p_vida * min(vivos, self.N) / self.N
                + self.p_paso * suma_q / self.N
                + self.p_fin * (1.0 if completado else 0.0))


# ======================================================================
if __name__ == '__main__':
    from p1_comun import cargar_config
    for f in ('A', 'B'):
        Q = Calidad(cargar_config(), f)
        print('=== Formulación {} ==='.format(f))
        print('d (m)   C_sep      |θ| (°)  C_ori      |e| (m)   C_car')
        ds = [0.10, 0.12, 0.16, 0.20, 0.325, 0.45, 0.55, 0.70, 0.90, 1.0]
        ts = [0, 10, 15, 18, 20, 23.3, 30, 35, 40, 45]
        es = [0, 0.02, 0.045, 0.06, 0.08, 0.10, 0.125, 0.15, 0.2, 0.3]
        for d, t, e in zip(ds, ts, es):
            print('{:5.3f}   {:4.2f}       {:5.1f}    {:4.2f}       {:5.3f}    {:4.2f}'.format(
                d, Q.c_sep(d), t, Q.c_ori(t), e, Q.c_car(e)))
        N = Q.N
        print('Ejemplos (N = {}, q = 1 en los pasos vividos): perfecto = {:.1f};  falla a los 2 s '
              '(4 pasos) = {:.1f};  a los 8 s (14 pasos) = {:.1f};  sobrevive sin hacerlo bien '
              '(q = 0) = {:.1f};  casi perfecto y falla al final = {:.1f}\n'.format(
                  N, Q.total(N, N, True), Q.total(3, 4, False), Q.total(13, 14, False),
                  Q.total(0, N, True), Q.total(N - 2, N - 1, False)))
