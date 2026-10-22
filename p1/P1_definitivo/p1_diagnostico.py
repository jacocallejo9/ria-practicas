#!/usr/bin/env python3
"""
Práctica 1 — Diagnóstico del simulador y del detector de blobs.

Mide, con la escena quieta (tras reiniciar, el guía espera mientras el
seguidor está lejos), lo que condiciona el periodo de control:

  1. Cuánto tarda el simulador en obedecer tras sim/reset_simulation.
  2. Cada cuánto cambia de verdad el blob (el puente lo publica a 5 Hz,
     pero puede repetir el mismo valor si el simulador no ha procesado
     una imagen nueva; en RoboboSim frame_timestamp vale 0 y no ayuda).
  3. El retraso entre el movimiento real y lo que se ve en el blob: el
     seguidor gira despacio en el sitio y se compara la x del blob con la
     que corresponde a su rotación real en cada instante.

    python3 p1_diagnostico.py
"""

import math
import time

import numpy as np

from p1_comun import cargar_config, desviacion
from p1_seguidor_env import RoboboSeguidorEnv

cfg = cargar_config()
env = RoboboSeguidorEnv(config=cfg, modo='protocolo')
try:
    # 1. Respuesta tras el reinicio
    env._detener()
    env.cliente.reiniciar()
    t0 = time.time()
    yaw0 = env.cliente.seguidor()[2]
    while True:
        env._aplicar(0.0, 0.3, 0.2)
        if abs(env.cliente.seguidor()[2] - yaw0) > 0.3:
            break
        if time.time() - t0 > 60:
            raise SystemExit('60 s sin respuesta (¿ventana del simulador sin foco?)')
    env._detener()
    print('1. Tras reset_simulation el seguidor obedece a los {:.1f} s'.format(time.time() - t0))
    env.colocar_tilt()
    env.colocar_seguidor_detras(0.6, en_linea=False)
    env._esperar(1.0)
    env.esperar_simulador_activo()

    # 2 y 3. Giro en el sitio a un lado y a otro, unos ±17° alrededor de la
    # pelota. Ojo: por debajo de ~5 unidades de rueda el robot simulado no
    # se mueve (zona muerta), así que el giro no puede ser muy lento.
    w_real = 0.6
    registros = []        # (t, x_blob, size, yaw_sim) en cada mensaje de blobs
    t0 = time.time()
    n = env._n_blob
    w_ant = None
    while time.time() - t0 < 8.0:
        t = time.time() - t0 + 0.25
        w = w_real if (t % 1.0) < 0.5 else -w_real
        if w != w_ant:                    # la orden se publica al cambiar; el
            env._publicar_velocidad(*env.orden_cmd(0.0, w))   # entorno la mantiene
            w_ant = w
        env._esperar(0.01)
        if env._n_blob != n:
            n = env._n_blob
            sx, sz, yaw, _ = env.cliente.seguidor()
            b = env._blob or (math.nan, math.nan, 0, 0)
            registros.append((time.time() - t0, b[0], b[2], yaw, sx, sz))
    env._detener()
    r = np.array(registros, dtype=float)
    giro = np.ptp(np.degrees(np.unwrap(np.radians(r[:, 3]))))
    if giro < 2.0:
        raise SystemExit('El seguidor no ha girado ({:.1f}°): el simulador se ha congelado '
                         'durante la medida. Repite con su ventana en primer plano.'.format(giro))
    print('   (el seguidor ha girado {:.1f}° en total)'.format(giro))
    cambios = np.sum(np.abs(np.diff(r[:, 1])) > 1e-6) + np.sum(np.abs(np.diff(r[:, 2])) > 1e-6)
    distintos = [0] + [i for i in range(1, len(r))
                       if r[i, 1] != r[i - 1, 1] or r[i, 2] != r[i - 1, 2]]
    print('2. {} mensajes de blobs en {:.1f} s ({:.1f} Hz); el valor cambia en {} de ellos '
          '({:.1f} Hz de imágenes nuevas)'.format(len(r), r[-1, 0], len(r) / r[-1, 0],
                                                len(distintos) - 1, (len(distintos) - 1) / r[-1, 0]))

    # x esperada a partir de la rotación real (modelo de cámara de la config)
    pelota = env.cliente.pelota()
    tan_semi = math.tan(math.radians(cfg['camara']['fov_horizontal_deg']) / 2)
    th = np.array([desviacion(np.array([q[4], q[5]]), env.rumbo(q[3]), pelota) for q in r])
    x_esp = 50 + 50 * np.tan(th) / tan_semi
    x_obs = r[:, 1]
    mejor = None
    for k in range(0, 15):                       # retraso en mensajes (0,2 s cada uno)
        a, b = x_obs[k:], x_esp[:len(x_esp) - k]
        ok = ~np.isnan(a)
        if ok.sum() > 5:
            err = float(np.mean(np.abs(a[ok] - b[ok])))
            if mejor is None or err < mejor[1]:
                mejor = (k, err)
    if mejor:
        print('3. La x del blob va retrasada ~{:.1f} s respecto a la rotación real '
              '(error medio {:.1f} unidades de x con ese retraso)'.format(mejor[0] * 0.2, mejor[1]))
    print('   (t, x_blob, x_esperada):')
    for i in range(0, len(r), max(1, len(r) // 20)):
        print('   {:5.2f}  {:6.1f}  {:6.1f}'.format(r[i, 0], x_obs[i], x_esp[i]))
finally:
    env.close()
