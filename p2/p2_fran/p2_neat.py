#!/usr/bin/env python3
"""
Práctica 2 — Evolución de un controlador con NEAT-Python sobre el entorno
de la Práctica 1 (RoboboSeguidorEnv).

Reutiliza de la P1, SIN MODIFICARLOS: p1_seguidor_env.py (entorno),
p1_comun.py, p1_simulador.py, p1_recompensa.py (la importa el entorno; su
recompensa se ignora) y p1_config.json (calibración).

--------------------------------------------------------------------
PUESTA EN MARCHA (igual que en la P1)
--------------------------------------------------------------------
  RoboboSim en Follow City, guía configurado y Apply (C1 = Mode 1, 10-10;
  C2 = Mode 1, 10-40; C3 = Mode 2, 10-40), ventana en primer plano.
  Terminal 1 (puente, sólo el módulo de blobs):
    source /root/robobo_ws/install/setup.bash
    ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal -p modules:="['blob']"
  Terminal 2 (ejecución): cd a esta carpeta y source /root/robobo_ws/install/setup.bash

  python3 p2_calidad.py                                   # tabla de la calidad
  python3 p2_neat.py --manual   --etiqueta C1             # referencia: controlador a mano
  python3 p2_neat.py --medir                              # tiempos y presupuesto
  python3 p2_neat.py --evolucionar --etiqueta C1 --semilla 1
  python3 p2_neat.py --reanudar    --etiqueta C1 --semilla 1          # tras un corte
  python3 p2_neat.py --reevaluar   --etiqueta C1 --semilla 1          # elige el genoma final
  python3 p2_neat.py --ejecutar    --etiqueta C1 --semilla 1          # vídeo
  # Evolución incremental: C2 partiendo de la población final de C1
  python3 p2_neat.py --evolucionar --etiqueta C2 --semilla 1 --generaciones 15 \\
          --desde p2_runs/C1_A_s1
  python3 p2_figuras.py --run p2_runs/C1_A_s1 --run p2_runs/C1_A_s2 \\
          --referencia p2_runs/referencia_C1_A.json --titulo "C1" --salida figuras/p2_C1
  # Protocolo del anexo B (120 s; C4 con la métrica de paradas): p2_protocolo.py
  python3 p2_protocolo.py --evaluar --run p2_runs/C1_A_s1 --configuracion C1
  python3 p2_protocolo.py p2_eval/*.csv

--------------------------------------------------------------------
DECISIONES DE DISEÑO (para la memoria)
--------------------------------------------------------------------
  Entradas de la red (6), preprocesadas a partir de la observación de la P1
  (11 componentes, sólo sensores del seguidor). Todas valen ~0 cuando el
  seguimiento es bueno, para que la salida de una red recién creada dependa
  sobre todo de sus sesgos y no salga saturada (tanh de NEAT = tanh(2,5·z)):
    err_dist    (d_estimada − 0,325) / 0,125, recortada a [−2, 2]: 0 en el
                centro de la banda, ±1 en sus bordes. (La P1 daba d/0,90.)
                d_estimada pasa por FiltroDistancia: si estando cerca la
                estimación salta hacia arriba (pelota cortada por el borde
                inferior de la imagen), se toma 0,15 m ("pegado") y
                vel_dist y vel_guia valen 0.
    x           x del blob en [−1, 1] (la última vista si no se ve).
    vel_dist    derivada de la distancia / v_max.
    vel_guia    velocidad estimada del guía / v_max.
    sin_pelota  tiempo sin ver la pelota / 3 s: 0 mientras se ve. Sustituye
                a 'visible', que valía 1 casi siempre y actuaba como un
                segundo sesgo aleatorio en cada salida.
    ir_frontal  proximidad del IR frontal en [0, 1].
  Acción: 2 salidas tanh en [−1, 1] (v, w); el entorno las reescala a
    velocidades reales exactamente como en la P1.
  Evaluación: 1 episodio por individuo, de N = 100 pasos (~60 s), que en C1
    incluye la primera curva. Siempre con reinicio de escena (como el anexo
    B.2): todos los individuos de una generación parten de lo mismo y se
    comparan en igualdad de condiciones. Modo 'entrenamiento' del entorno:
    además de los fallos del protocolo corta por salirse de la calzada y
    por adelantar al guía.
  Si el simulador se congela, el episodio se repite. Si cae el puente, la
    evolución se detiene (NO se asigna 0 al individuo): se relanza el
    puente y se usa --reanudar.
"""

import argparse
import csv
import glob
import json
import math
import os
import pickle
import re
import shutil
import time

from itertools import count

import numpy as np
import neat

from p1_comun import cargar_config
from p2_calidad import Calidad
from p2_registro import RegistroGeneracion, CONTADOR, ESTADO

P1_DIR = os.path.dirname(os.path.abspath(__file__))
NOMBRES_SALIDA = ['v', 'w']
GUIA = {'C1': 'Mode 1, 10-10 (crucero constante)', 'C2': 'Mode 1, 10-40 (crucero)',
        'C3': 'Mode 2, 10-40 (variable)', 'C4': 'Mode 2 + paradas, 10-40'}

# Versión de las entradas. La 1 es la de las primeras evoluciones
# (p2_runs/C1_A_s1*): seis componentes de la observación tal cual. Se
# conserva para poder reevaluar y ejecutar esos genomas.
ENTRADAS = {
    1: ['visible', 'x', 'distancia', 'vel_dist', 'vel_guia', 'ir_frontal'],
    2: ['err_dist', 'x', 'vel_dist', 'vel_guia', 'sin_pelota', 'ir_frontal'],
}
VERSION_ENTRADAS = 2
OBS_IDX_V1 = [0, 1, 3, 5, 6, 8]


class ErrorInfraestructura(Exception):
    """Fallo del simulador o del puente: no es culpa del individuo."""


# =====================================================================
# Preprocesado de la observación (declarado en la memoria)
# =====================================================================

class FiltroDistancia:
    """Corrige la distancia estimada del blob cuando el seguidor está pegado.

    Por debajo de ~0,20 m la pelota se sale por el borde inferior de la
    imagen: el blob se ve más pequeño y la distancia estimada SALTA (en
    RoboboSim, de 0,32 a 1,01 m con 0,17 m reales, y la velocidad estimada
    del guía a +1,9 m/s). Un controlador que se fíe acelera y choca.

    Sólo usa sensores del robot (la lectura anterior): si con la última
    distancia fiable por debajo de CERCA la estimación sube más de SALTO en
    un paso (imposible físicamente: el guía va como mucho a ~0,22 m/s y un
    paso dura ~0,4-0,6 s), se considera "pegado" y se devuelve D_PEGADO, con
    las velocidades derivadas a 0, hasta que vuelva una lectura plausible.
    Mientras está pegado y no se ve la pelota, sigue pegado.
    """
    CERCA, SALTO, D_PEGADO = 0.40, 0.25, 0.15

    def __init__(self):
        self.d_fiable, self.pegado = None, False

    def __call__(self, visible, d):
        """-> (distancia corregida, pegado)."""
        if not visible:
            return (self.D_PEGADO, True) if self.pegado else (d, False)
        if self.d_fiable is not None and self.d_fiable < self.CERCA \
                and d - self.d_fiable > self.SALTO:
            self.pegado = True
            return self.D_PEGADO, True
        self.d_fiable, self.pegado = d, False
        return d, False


class Preprocesado:
    """Observación del entorno (11) -> entradas de la red (6), versión 2.

    Tiene estado (FiltroDistancia): cada política usa una copia nueva
    (nuevo()), para que no se mezcle la memoria de un individuo con la del
    siguiente.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.perdido = cfg['protocolo']['perdido_m']
        lo, hi = cfg['protocolo']['banda_m']
        self.d_ref, self.semi = 0.5 * (lo + hi), 0.5 * (hi - lo)
        self.filtro = FiltroDistancia()

    def nuevo(self):
        return Preprocesado(self.cfg)

    def distancia(self, obs):
        """(distancia corregida en m, pegado) a partir de la observación."""
        return self.filtro(obs[0] > 0.5, float(obs[3]) * self.perdido)

    def __call__(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        d, pegado = self.distancia(o)
        err_d = np.clip((d - self.d_ref) / self.semi, -2.0, 2.0)
        vel_d, vel_g = (0.0, 0.0) if pegado else (o[5], o[6])
        return np.array([err_d, o[1], vel_d, vel_g, o[7], o[8]])


def preprocesador(cfg, version=VERSION_ENTRADAS):
    """Función observación del entorno (11) -> entradas de la red (6)."""
    if version == 1:
        return lambda obs: np.asarray(obs, dtype=np.float64)[OBS_IDX_V1]
    return Preprocesado(cfg)


def version_de_meta(meta):
    return int(meta.get('entradas_version', 1))


def politica_de_genoma(genoma, config, pre):
    """Función observación completa (11) -> acción (2) a partir de un genoma."""
    red = neat.nn.FeedForwardNetwork.create(genoma, config)
    pre = pre.nuevo() if hasattr(pre, 'nuevo') else pre
    return lambda obs: np.clip(np.array(red.activate(pre(obs)), dtype=np.float32), -1.0, 1.0)


def controlador_manual(cfg):
    """Controlador proporcional programado a mano (referencia del reto).

    Usa la misma información que la red (la distancia ya corregida por
    FiltroDistancia):
      v = K_D · (d − d_para), recortada a [v_min, v_max]. d_para = 0,28 m de
          distancia estimada: de cerca la estimación se pasa (0,28 con
          0,20 m reales), así que parar en 0,20 estimados lo dejaba pegado
          al guía. En régimen, d = d_para + v_guia/K_D: ~0,34 m en C1 (guía a
          ~0,05 m/s) y ~0,55 m a la velocidad máxima de C2 (~0,22 m/s).
      w = −K_X · x: gira hacia la pelota (x > 0, a la derecha -> w < 0, el
          mismo signo que las políticas de la P1).
      Pegado al guía (pelota cortada por abajo): parado.
      Sin pelota: parado y girando hacia el lado por el que se fue.
    """
    a = cfg['accion']
    v_min, v_max = a['v_min'], a['v_max']
    d_para, K_D, K_X = 0.28, 0.8, 0.8
    pre = Preprocesado(cfg)

    def f(obs):
        d, pegado = pre.distancia(obs)
        if pegado:
            return np.array([-1.0, float(np.clip(-K_X * obs[1], -1.0, 1.0))], dtype=np.float32)
        if obs[0] < 0.5:
            return np.array([-1.0, -0.5 * float(np.sign(obs[1]))], dtype=np.float32)
        v = float(np.clip(K_D * (d - d_para), max(v_min, 0.0), v_max))
        return np.array([2.0 * (v - v_min) / (v_max - v_min) - 1.0,
                         float(np.clip(-K_X * obs[1], -1.0, 1.0))], dtype=np.float32)
    return f


# =====================================================================
# Utilidades
# =====================================================================

def asegurar_linea(cfg):
    """El entorno carga la línea central de un .npy; si sólo hay .csv, lo convierte."""
    npy = cfg['carril'].get('archivo_linea_central', '')
    if npy and not os.path.exists(npy):
        csv_ = npy.replace('.npy', '.csv')
        if os.path.exists(csv_):
            np.save(npy, np.loadtxt(csv_, delimiter=',', skiprows=1))
            print('Línea central: {} -> {}'.format(csv_, npy))
        else:
            print('AVISO: no hay línea central: no se cortará por salirse de la calzada '
                  'y C_car valdrá 1.')


OPCIONES = {'detalle': False}

# Comprobación del signo de la x del blob. Convención de la P1: x > 0 cuando
# la pelota está a la derecha, que es cuando la desviación real (simulador)
# es positiva. En la primera sesión de la P2 llegó invertida (se corrigió
# con camara.x_semiancho < 0 en p1_config.json). Se cuentan los pasos con
# la pelota visible y claramente descentrada.
SIGNO_X = {'acuerdo': 0, 'total': 0}


def anotar_signo_x(info):
    d = info.get('desviacion_deg')
    if info.get('visible') and d is not None and abs(d) > 3.0 and abs(info.get('x', 0.0)) > 0.05:
        SIGNO_X['total'] += 1
        SIGNO_X['acuerdo'] += int(info['x'] * d > 0)


def comprobar_signo_x(minimo=15, abortar=False):
    """Avisa (o detiene) si la x del blob va al revés que la desviación."""
    n, a = SIGNO_X['total'], SIGNO_X['acuerdo']
    if n < minimo:
        print('Signo de la x del blob: sin datos suficientes ({} pasos descentrados).'.format(n))
        return None                  # se vuelve a mirar en la siguiente generación
    if a / n >= 0.7:
        print('Signo de la x del blob: correcto ({}/{} pasos).'.format(a, n))
        return True
    msg = ('\n*** La x del blob va AL REVÉS que la desviación real ({}/{} pasos de acuerdo). '
           'Cambia el signo de camara.x_semiancho en p1_config.json. ***'.format(a, n))
    if abortar:
        raise SystemExit(msg)
    print(msg)
    return False


def crear_env(cfg, n_pasos):
    from p1_seguidor_env import RoboboSeguidorEnv
    return RoboboSeguidorEnv(config=cfg, modo='entrenamiento', pasos_max=n_pasos,
                             verbose=OPCIONES['detalle'])


def directorio_run(args):
    return args.salida or os.path.join('p2_runs', '{}_{}_s{}'.format(
        args.etiqueta, args.formulacion, args.semilla))


def cargar_neat(ruta_ini, n_entradas=None):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, ruta_ini)
    n_in = config.genome_config.num_inputs
    n_entradas = len(ENTRADAS[VERSION_ENTRADAS]) if n_entradas is None else n_entradas
    if n_in != n_entradas:
        raise SystemExit('num_inputs = {} en {} pero la red recibe {} entradas.'.format(
            n_in, ruta_ini, n_entradas))
    return config


def ultimo_checkpoint(run):
    chk = sorted(glob.glob(os.path.join(run, 'chk', 'p2-*')),
                 key=lambda r: int(r.rsplit('-', 1)[1]))
    if not chk:
        raise SystemExit('No hay puntos de control en {}/chk'.format(run))
    return chk[-1]


def leer_meta(run):
    with open(os.path.join(run, 'meta.json')) as f:
        return json.load(f)


# =====================================================================
# Evaluación de individuos
# =====================================================================

class Evaluador:
    """Ejecuta episodios del entorno y calcula la calidad (p2_calidad.py)."""

    CAMPOS_LOG = ['generacion', 'genoma', 'episodio', 'calidad', 'pasos', 'causa',
                  'completado', 'c_sep', 'c_ori', 'c_car', 'segundos', 'reinicio']

    def __init__(self, env, calidad, episodios=1, agregar='media', ruta_log=None,
                 reinicio='escena', semilla=None):
        self.env, self.calidad = env, calidad
        self.reinicio = reinicio
        self.semilla = semilla        # la del entorno (ruido de 'colocar'), sólo en el 1.er reset
        self.N = calidad.N
        self.k, self.agregar = episodios, agregar
        self.ruta_log = ruta_log
        if ruta_log and not os.path.exists(ruta_log):
            os.makedirs(os.path.dirname(ruta_log) or '.', exist_ok=True)
            with open(ruta_log, 'w', newline='') as f:
                csv.writer(f).writerow(self.CAMPOS_LOG)

    # Arranque válido (estado tras el reinicio de escena y la acción de
    # arranque, verdad del terreno). En las evoluciones de C1 un 3-12 % de
    # los episodios terminaba en 1-3 pasos (fuera_calzada en el paso 1, que
    # ninguna política puede provocar en 0,4 s), también con el controlador
    # a mano: el reinicio a veces deja al seguidor mal colocado. Eso es un
    # fallo de la infraestructura, no del individuo, y destruía a la élite
    # (genoma 130 de C1_v3_s3: F = 83 y en la generación siguiente 0).
    SEP_INICIO = (0.15, 0.60)
    DESV_INICIO_DEG = 30.0
    PASOS_SOSPECHOSOS = 2
    ESTADO_ARRANQUES = {'ruta': None}

    def arranque_valido(self, info):
        if 'separacion' not in info:          # sin simulador (pruebas): no se comprueba
            return True
        return (info.get('en_calzada', True) is not False
                and self.SEP_INICIO[0] <= info['separacion'] <= self.SEP_INICIO[1]
                and abs(info['desviacion_deg']) <= self.DESV_INICIO_DEG)

    def anotar_arranque(self, info0, valido, pasos, causa, accion):
        ruta = self.ESTADO_ARRANQUES['ruta']
        if not ruta:
            return
        nuevo = not os.path.exists(ruta)
        with open(ruta, 'a', newline='') as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(['generacion', 'sep0', 'desv0', 'lat0', 'calzada0', 'valido',
                            'pasos', 'causa', 'accion'])
            w.writerow([ESTADO['generacion'], round(info0.get('separacion', float('nan')), 3),
                        round(info0.get('desviacion_deg', float('nan')), 1),
                        round(info0.get('error_lateral', float('nan')), 3),
                        int(info0.get('en_calzada', True) is not False), int(valido),
                        pasos, causa, accion])

    def episodio(self, politica):
        """Un episodio completo -> dict.

        Se repite (sin contar para el individuo) si el simulador se congela,
        si el arranque deja al seguidor en un estado no válido, o si el
        episodio falla en los PASOS_SOSPECHOSOS primeros pasos (una sola
        repetición: si vuelve a fallar igual, cuenta; así un individuo que
        de verdad choca nada más empezar no sale beneficiado).
        """
        repetido_corto = False
        for _ in range(6):
            t0 = time.time()
            try:
                obs, info = self.env.reset(seed=self.semilla,
                                           options={'reinicio': self.reinicio})
                self.semilla = None
                info0 = dict(info)
                if not self.arranque_valido(info0):
                    self.anotar_arranque(info0, False, 0, '', 'se repite el reinicio')
                    print('  [arranque no válido: sep {:.2f} m, desv {:+.0f}°, calzada {}: '
                          'se repite el reinicio]'.format(info0['separacion'],
                                                          info0['desviacion_deg'],
                                                          info0.get('en_calzada')))
                    continue
                suma, vivos, causa = 0.0, 0, ''
                comp = np.zeros(3)
                for _ in range(self.N):
                    obs, _, term, trunc, info = self.env.step(politica(obs))
                    vivos += 1
                    causa = info['causa']
                    if causa == 'congelado':
                        break
                    anotar_signo_x(info)
                    if causa in ('', 'tiempo'):          # el paso en que se falla no suma
                        q, t = self.calidad.paso(info)
                        suma += q
                        comp += t
                    if term or trunc:
                        break
            except RuntimeError as e:
                raise ErrorInfraestructura(str(e))
            if causa == 'congelado':
                print('  [simulador congelado: se repite la evaluación]')
                continue
            if causa not in ('', 'tiempo') and vivos <= self.PASOS_SOSPECHOSOS \
                    and not repetido_corto:
                repetido_corto = True
                self.anotar_arranque(info0, True, vivos, causa, 'fallo inmediato: se repite')
                print('  [{} en el paso {}: fallo del arranque o del individuo; se repite una '
                      'vez]'.format(causa, vivos))
                continue
            self.anotar_arranque(info0, True, vivos, causa, '')
            completado = causa == 'tiempo'
            sin_fallo = vivos if causa in ('', 'tiempo') else vivos - 1
            CONTADOR['pasos'] += vivos
            CONTADOR['episodios'] += 1
            CONTADOR['completados'] += int(completado)
            return {'calidad': float(self.calidad.total(suma, sin_fallo, completado)), 'pasos': vivos,
                    'causa': causa or 'tiempo', 'completado': completado,
                    'c_sep': comp[0] / vivos, 'c_ori': comp[1] / vivos, 'c_car': comp[2] / vivos,
                    'segundos': time.time() - t0, 'reinicio': self.reinicio}
        raise ErrorInfraestructura('6 reinicios seguidos congelados o con un arranque no válido.')

    def evaluar(self, politica, id_genoma=-1, k=None, registrar=True):
        """Calidad agregada de k episodios y lista de resultados."""
        res = []
        for ep in range(k or self.k):
            r = self.episodio(politica)
            res.append(r)
            if registrar and self.ruta_log:
                with open(self.ruta_log, 'a', newline='') as f:
                    csv.writer(f).writerow([ESTADO['generacion'], id_genoma, ep, round(r['calidad'], 3),
                                            r['pasos'], r['causa'], int(r['completado']),
                                            round(r['c_sep'], 3), round(r['c_ori'], 3),
                                            round(r['c_car'], 3), round(r['segundos'], 1), r['reinicio']])
        cal = [r['calidad'] for r in res]
        valor = float(np.min(cal) if self.agregar == 'min' else np.mean(cal))
        if not math.isfinite(valor):
            valor = 0.0
        return valor, res


def comprobar_centinela(ev, ruta_csv, intentos=2):
    """Comprueba que el simulador sigue sano antes de evaluar una generación.

    El centinela es el controlador a mano, que en un simulador sano completa
    C1 (3/3 en la referencia). En C1_v2_s1 los mismos genomas que
    completaban el episodio generación tras generación empezaron a fallar
    tras ~1,5-2 h de evolución. Si el centinela no completa ninguno de sus
    intentos, se detiene la evolución ANTES de evaluar la generación (no se
    pierde nada): se reinicia RoboboSim y el puente y se sigue con
    --reanudar. Sus pasos no cuentan en el coste de la evolución, pero su
    tiempo sí entra en seg_generacion (~1 min cada pocas generaciones).
    """
    gen = ESTADO['generacion']
    for intento in range(1, intentos + 1):
        guardado = dict(CONTADOR)
        valor, res = ev.evaluar(controlador_manual(ev.env.cfg), k=1, registrar=False)
        CONTADOR.update(guardado)
        r = res[0]
        nuevo = not os.path.exists(ruta_csv)
        with open(ruta_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(['generacion', 'intento', 'calidad', 'pasos', 'causa', 'hora'])
            w.writerow([gen, intento, round(valor, 2), r['pasos'], r['causa'],
                        time.strftime('%Y-%m-%d %H:%M:%S')])
        print('  [centinela gen {}] controlador a mano: F = {:.1f}, {} pasos, {}'.format(
            gen, valor, r['pasos'], r['causa']))
        if r['completado']:
            return
    raise SystemExit(
        '\n*** El controlador a mano (centinela) no completa C1 en {} intentos: el simulador '
        'parece degradado o mal configurado. La evolución se detiene sin evaluar la generación {}. '
        'Reinicia RoboboSim y el puente, comprueba el guía (C1) y sigue con --reanudar. ***'.format(
            intentos, gen))


def crear_eval_genomes(ev, pre, centinela=0, ruta_centinela=None):
    def eval_genomes(genomas, config):
        SIGNO_X.update(acuerdo=0, total=0)          # se cuenta por generación
        if centinela and (not ESTADO.get('centinela_hecho')
                          or ESTADO['generacion'] % centinela == 0):
            comprobar_centinela(ev, ruta_centinela)
            ESTADO['centinela_hecho'] = True
        for n, (gid, g) in enumerate(genomas, 1):
            valor, res = ev.evaluar(politica_de_genoma(g, config, pre), gid)
            g.fitness = valor
            CONTADOR['evaluaciones'] += 1
            r = res[-1]
            print('  gen {:3d}  ind {:2d}/{:2d} (id {:4d})  F = {:6.2f}  {:3d} pasos  {:13s} '
                  'sep {:.2f} ori {:.2f} car {:.2f}'.format(
                      ESTADO['generacion'], n, len(genomas), gid, valor, r['pasos'], r['causa'],
                      r['c_sep'], r['c_ori'], r['c_car']))
        # En CADA generación (en C1_v2_s1 la x se invirtió a mitad de la
        # evolución, en la generación 17, y hundió a toda la población): con
        # la x invertida se detiene, antes de guardar el punto de control de
        # esa generación. Se arregla el simulador y se sigue con --reanudar.
        comprobar_signo_x(abortar=True)
    return eval_genomes


# =====================================================================
# --evolucionar / --reanudar
# =====================================================================

def preparar_config(ruta_ini, destino, semilla):
    """Copia el .ini a la carpeta de la evolución con la semilla indicada."""
    with open(ruta_ini, encoding='utf-8') as f:
        txt = f.read()
    if re.search(r'^\s*seed\s*=', txt, re.M):
        txt = re.sub(r'^(\s*seed\s*=).*$', r'\g<1> {}'.format(semilla), txt, flags=re.M)
    else:
        txt = txt.replace('[NEAT]\n', '[NEAT]\nseed = {}\n'.format(semilla), 1)
    with open(destino, 'w', encoding='utf-8') as f:
        f.write(txt)


def reparar_indexador_nodos(poblacion):
    """Pone el contador de nodos nuevos por encima de todos los existentes.

    NEAT-Python 2.0, al restaurar un punto de control con una config nueva
    (--desde), deja node_indexer a None y lo inicializa con el máximo del
    PRIMER genoma que muta: si otro genoma ya tiene ese número de nodo,
    falla 'assert new_id not in node_dict' al reproducir. Así se cortaron
    C2_inc_s2 (contador en 7, nodos hasta 39) y C2_inc_s3; sus puntos de
    control guardan el contador malo, así que se repara también al reanudar.
    """
    gc = poblacion.config.genome_config
    nodos = [k for g in poblacion.population.values() for k in g.nodes]
    for s in poblacion.species.species.values():
        if s.representative is not None:
            nodos.extend(s.representative.nodes)
    siguiente = max(nodos + [gc.num_outputs - 1]) + 1
    if gc.node_indexer is not None:
        actual = next(gc.node_indexer)        # consume uno: no importa, sólo saltamos ids
        siguiente = max(siguiente, actual)
    gc.node_indexer = count(siguiente)


def poblacion_desde(origen, ini):
    """Evolución incremental: la población del último punto de control de
    otra evolución (p. ej. la final de C1) continúa con otra configuración
    del guía, con el .ini de la nueva carpeta.

    origen puede ser la carpeta (se usa su último punto de control) o un
    punto de control concreto (.../chk/p2-k): la población final de una
    evolución puede ser peor que la de generaciones anteriores (en
    C1_v3_s2 el mejor de la última generación sacó 18,6 y el genoma final,
    de la 21, 93,5). La nueva fase cuenta sus generaciones desde 0."""
    if os.path.isfile(origen):
        chk = origen
        run_origen = os.path.dirname(os.path.dirname(os.path.abspath(chk)))
    else:
        run_origen = origen
        chk = ultimo_checkpoint(origen)
    meta = leer_meta(run_origen)
    if version_de_meta(meta) != VERSION_ENTRADAS:
        raise SystemExit('{} usa otras entradas (versión {}): no se puede continuar.'.format(
            run_origen, version_de_meta(meta)))
    print('Evolución incremental desde {}'.format(chk))
    poblacion = neat.Checkpointer.restore_checkpoint(chk, new_config=cargar_neat(ini))
    # Generaciones desde 0 en la nueva carpeta (puntos de control, CSV y
    # --reanudar cuadran) y estancamiento contado desde el inicio de la fase.
    poblacion.generation = 0
    for s in poblacion.species.species.values():
        s.created = 0
        s.last_improved = 0
        s.fitness_history = []
    reparar_indexador_nodos(poblacion)
    return poblacion, chk


def modo_evolucionar(args, reanudar=False):
    run = directorio_run(args)
    # Antes de escribir nada: no pisar una evolución anterior (ni su
    # meta.json, que dice con qué entradas se evolucionaron sus genomas).
    if not reanudar and (os.path.exists(os.path.join(run, 'evolucion.csv'))
                         or glob.glob(os.path.join(run, 'chk', 'p2-*'))):
        raise SystemExit('{} ya tiene una evolución: usa --reanudar, otra --semilla, otra '
                         '--salida o renombra la carpeta (p. ej. a {}_intento2).'.format(run, run))
    os.makedirs(os.path.join(run, 'chk'), exist_ok=True)
    ini = os.path.join(run, 'config-p2.ini')
    if not reanudar or not os.path.exists(ini):     # una evolución nueva, siempre el .ini actual
        preparar_config(args.config_ini, ini, args.semilla)
    config = cargar_neat(ini)
    if reanudar:
        meta = leer_meta(run)
        if version_de_meta(meta) != VERSION_ENTRADAS:
            raise SystemExit('{} es una evolución con las entradas antiguas (versión {}); '
                             'no se puede reanudar con este código.'.format(run, version_de_meta(meta)))
        args.pasos, args.reinicio = meta['n_pasos'], meta['reinicio']
        args.episodios_ind, args.agregar = meta['episodios_por_individuo'], meta['agregar']
        args.formulacion = meta['formulacion']
    cfg = cargar_config(args.config_p1 if not reanudar else os.path.join(run, 'p1_config.json'))
    asegurar_linea(cfg)
    if not reanudar:
        shutil.copy(args.config_p1, os.path.join(run, 'p1_config.json'))
        meta = {'etiqueta': args.etiqueta, 'formulacion': args.formulacion, 'semilla': args.semilla,
                'n_pasos': args.pasos, 'episodios_por_individuo': args.episodios_ind,
                'agregar': args.agregar, 'reinicio': args.reinicio,
                'entradas_version': VERSION_ENTRADAS, 'entradas': ENTRADAS[VERSION_ENTRADAS],
                'salidas': NOMBRES_SALIDA, 'guia': GUIA.get(args.etiqueta, ''),
                'desde': args.desde, 'multiplicador_simulador': args.multiplicador}
        with open(os.path.join(run, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    if not args.sin_pausa:
        input('\nConfigura el guía en RoboboSim como {} -> {}, pulsa Apply, deja la ventana '
              'en primer plano y pulsa Enter... '.format(args.etiqueta, GUIA.get(args.etiqueta, '?')))

    ya = 0
    ruta_csv = os.path.join(run, 'evolucion.csv')
    if reanudar:
        chk = ultimo_checkpoint(run)
        poblacion = neat.Checkpointer.restore_checkpoint(chk)
        reparar_indexador_nodos(poblacion)
        if os.path.exists(ruta_csv):
            with open(ruta_csv) as f:
                ya = max(sum(1 for _ in f) - 1, 0)
        n_chk = int(chk.rsplit('-', 1)[1])
        if n_chk != ya:
            raise SystemExit('El último punto de control ({}) no corresponde a las {} generaciones '
                             'de evolucion.csv.'.format(chk, ya))
        # Las evaluaciones de una generación a medias (Ctrl-C, signo invertido)
        # se descartan: esa generación se repite entera.
        ruta_ev = os.path.join(run, 'evaluaciones.csv')
        if os.path.exists(ruta_ev):
            with open(ruta_ev, newline='') as f:
                filas = list(csv.reader(f))
            buenas = [filas[0]] + [r for r in filas[1:] if int(r[0]) < ya]
            if len(buenas) < len(filas):
                with open(ruta_ev, 'w', newline='') as f:
                    csv.writer(f).writerows(buenas)
                print('Descartadas {} evaluaciones de la generación {} a medias.'.format(
                    len(filas) - len(buenas), ya))
        print('Reanudando desde {} ({} generaciones ya hechas).'.format(chk, ya))
    else:
        if args.desde:
            poblacion, _ = poblacion_desde(args.desde, ini)
        else:
            poblacion = neat.Population(config)
    faltan = args.generaciones - ya
    if faltan <= 0:
        raise SystemExit('Ya están hechas las {} generaciones.'.format(args.generaciones))

    env = crear_env(cfg, args.pasos)
    ev = Evaluador(env, Calidad(cfg, args.formulacion, args.pasos), args.episodios_ind,
                   args.agregar, os.path.join(run, 'evaluaciones.csv'),
                   reinicio=args.reinicio, semilla=None if reanudar else args.semilla)
    Evaluador.ESTADO_ARRANQUES['ruta'] = os.path.join(run, 'arranques.csv')
    poblacion.add_reporter(neat.StdOutReporter(True))
    poblacion.add_reporter(neat.Checkpointer(generation_interval=1,
                                             filename_prefix=os.path.join(run, 'chk', 'p2-')))
    poblacion.add_reporter(RegistroGeneracion(ruta_csv, os.path.join(run, 'genomas')))
    print('\nEvolución {} · formulación {} · semilla {} · {} generaciones de {} individuos · '
          '{} pasos/episodio · reinicio {} -> {}'.format(
              args.etiqueta, args.formulacion, args.semilla, faltan, poblacion.config.pop_size,
              args.pasos, args.reinicio, run))
    try:
        mejor = poblacion.run(crear_eval_genomes(
            ev, preprocesador(cfg), args.centinela, os.path.join(run, 'centinela.csv')), faltan)
        with open(os.path.join(run, 'mejor_genoma.pkl'), 'wb') as f:
            pickle.dump(mejor, f)
        print('\nEvolución terminada. Mejor F registrada: {:.2f}. Siguiente paso: --reevaluar'.format(
            mejor.fitness))
    except ErrorInfraestructura as e:
        print('\n*** Evolución DETENIDA por un fallo de infraestructura: {}\n'
              '    Relanza el puente (y abre bien el simulador) y continúa con --reanudar. ***'.format(e))
    except KeyboardInterrupt:
        print('\nInterrumpida. Se puede continuar con --reanudar.')
    finally:
        env.close()


# =====================================================================
# --manual : referencia con el controlador a mano
# =====================================================================

def modo_manual(args):
    cfg = cargar_config(args.config_p1)
    asegurar_linea(cfg)
    env = crear_env(cfg, args.pasos)
    try:
        if not args.sin_pausa:
            input('Configura el guía como {} ({}), Apply y pulsa Enter... '.format(
                args.etiqueta, GUIA.get(args.etiqueta, '?')))
        ev = Evaluador(env, Calidad(cfg, args.formulacion, args.pasos), reinicio=args.reinicio)
        valor, res = ev.evaluar(controlador_manual(cfg), k=args.episodios, registrar=False)
        for i, r in enumerate(res, 1):
            print('episodio {}: F = {:.2f}  {}  {} pasos  sep {:.2f} ori {:.2f} car {:.2f}'.format(
                i, r['calidad'], r['causa'], r['pasos'], r['c_sep'], r['c_ori'], r['c_car']))
        os.makedirs('p2_runs', exist_ok=True)
        ruta = os.path.join('p2_runs', 'referencia_{}_{}.json'.format(args.etiqueta, args.formulacion))
        with open(ruta, 'w') as f:
            json.dump({'etiqueta': args.etiqueta, 'formulacion': args.formulacion,
                       'n_pasos': args.pasos, 'reinicio': args.reinicio,
                       'calidad_media': valor, 'calidades': [r['calidad'] for r in res],
                       'causas': [r['causa'] for r in res],
                       'completados': sum(r['completado'] for r in res)}, f, indent=2)
        print('\nCalidad media del controlador a mano: {:.2f} -> {}'.format(valor, ruta))
        comprobar_signo_x(minimo=5)
    finally:
        env.close()


# =====================================================================
# --medir : tiempos y presupuesto de cómputo
# =====================================================================

def modo_medir(args):
    cfg = cargar_config(args.config_p1)
    asegurar_linea(cfg)
    config = cargar_neat(args.config_ini)
    env = crear_env(cfg, args.pasos)
    try:
        t_reset = []
        for _ in range(2):
            t0 = time.time()
            env.reset(options={'reinicio': 'escena'})
            t_reset.append(time.time() - t0)
        print('Tiempo de reset (T_r): {} s'.format(', '.join('{:.1f}'.format(t) for t in t_reset)))
        ev = Evaluador(env, Calidad(cfg, args.formulacion, args.pasos))
        # Con el controlador a mano el episodio suele durar entero: mide el paso real.
        valor, res = ev.evaluar(controlador_manual(cfg), registrar=False)
        r = res[0]
        t_paso = (r['segundos'] - t_reset[-1]) / max(r['pasos'], 1)
        print('Controlador a mano: F = {:.2f}, {} pasos, {}, ~{:.2f} s/paso'.format(
            valor, r['pasos'], r['causa'], t_paso))
        comprobar_signo_x(minimo=5)
        T = args.pasos * (t_paso if r['pasos'] > 5 else 0.6)
        Tr = float(np.mean(t_reset))
        P, k, G = config.pop_size, args.episodios_ind, args.generaciones
        gen = P * k * (Tr + T)
        print('Episodio máximo T = {} pasos ≈ {:.0f} s'.format(args.pasos, T))
        print('Generación (máx.) = {} x {} x ({:.1f} + {:.0f}) = {:.1f} min'.format(
            P, k, Tr, T, gen / 60))
        print('Evolución de {} generaciones (máx.) = {:.1f} h  (en la práctica algo menos)'.format(
            G, gen * G / 3600))
        if gen * G > 6 * 3600:
            print('  AVISO: supera las 6 h recomendadas; baja pop_size, generaciones o pasos.')
    finally:
        env.close()


# =====================================================================
# --reevaluar : elegir el genoma final
# =====================================================================

def modo_reevaluar(args):
    run = directorio_run(args)
    meta = leer_meta(run)
    version = version_de_meta(meta)
    config = cargar_neat(os.path.join(run, 'config-p2.ini'), len(ENTRADAS[version]))
    cfg = cargar_config(os.path.join(run, 'p1_config.json'))
    asegurar_linea(cfg)
    pre = preprocesador(cfg, version)
    with open(os.path.join(run, 'evolucion.csv')) as f:
        filas = list(csv.DictReader(f))
    # Candidatos, sin repetir genoma: sobre todo los de mayor calidad
    # registrada (con un único episodio la calidad es ruidosa, por eso se
    # reevalúan) y, en las 2 últimas plazas, los mejores de las últimas
    # generaciones (población madura). Primero se ponía a los recientes: si
    # la evolución se hundía al final (C1_v2_s1_rescate) sólo se reevaluaba
    # a genomas malos.
    def ruta_gen(r):
        return os.path.join(run, 'genomas', 'mejor_gen{:03d}.pkl'.format(int(r['generacion'])))
    por_calidad = [ruta_gen(r) for r in sorted(filas, key=lambda r: -float(r['cal_mejor']))]
    recientes = [ruta_gen(r) for r in filas[::-1]]
    extra = os.path.join(run, 'mejor_genoma.pkl')
    candidatos, vistos = [], set()

    def anadir(ruta, limite):
        if len(candidatos) >= limite or not os.path.exists(ruta):
            return
        with open(ruta, 'rb') as f:
            g = pickle.load(f)
        if g.key not in vistos:
            vistos.add(g.key)
            candidatos.append((ruta, g))

    n_rec = min(2, max(args.candidatos - 1, 0))
    for ruta in [extra] + por_calidad:
        anadir(ruta, args.candidatos - n_rec)
    for ruta in recientes + por_calidad:
        anadir(ruta, args.candidatos)
    print('Reevaluando {} candidatos x {} episodios (formulación {}, reinicio de escena)'.format(
        len(candidatos), args.episodios, meta['formulacion']))
    env = crear_env(cfg, meta['n_pasos'])
    try:
        if not args.sin_pausa:
            input('Guía como {} ({}), Apply y Enter... '.format(
                meta['etiqueta'], GUIA.get(meta['etiqueta'], '?')))
        ev = Evaluador(env, Calidad(cfg, meta['formulacion'], meta['n_pasos']), reinicio='escena')
        tabla = []
        for ruta, g in candidatos:
            valor, res = ev.evaluar(politica_de_genoma(g, config, pre), g.key,
                                    k=args.episodios, registrar=False)
            comp = sum(r['completado'] for r in res)
            tabla.append((valor, comp, ruta, g, [r['calidad'] for r in res]))
            print('  {:28s} id {:4d}  F media {:6.2f}  completados {}/{}  (original {:.2f})'.format(
                os.path.basename(ruta), g.key, valor, comp, len(res), g.fitness))
        tabla.sort(key=lambda t: (t[1], t[0]), reverse=True)
        valor, comp, ruta, g, cals = tabla[0]
        with open(os.path.join(run, 'genoma_final.pkl'), 'wb') as f:
            pickle.dump(g, f)
        with open(os.path.join(run, 'genoma_final.json'), 'w') as f:
            json.dump({'origen': ruta, 'id': g.key, 'calidad_media': valor, 'completados': comp,
                       'calidades': cals, 'nodos_conexiones': list(g.size())}, f, indent=2)
        print('\nGenoma final: id {} ({}), F media {:.2f}, {}/{} completados -> {}/genoma_final.pkl'
              .format(g.key, os.path.basename(ruta), valor, comp, args.episodios, run))
    finally:
        env.close()


# =====================================================================
# --ejecutar : el genoma final sin fin (vídeos)
# =====================================================================

def modo_ejecutar(args):
    run = directorio_run(args)
    meta = leer_meta(run)
    version = version_de_meta(meta)
    config = cargar_neat(os.path.join(run, 'config-p2.ini'), len(ENTRADAS[version]))
    ruta = os.path.join(run, 'genoma_final.pkl')
    if not os.path.exists(ruta):
        ruta = os.path.join(run, 'mejor_genoma.pkl')
    with open(ruta, 'rb') as f:
        g = pickle.load(f)
    cfg = cargar_config(os.path.join(run, 'p1_config.json'))
    asegurar_linea(cfg)
    env = crear_env(cfg, 10 ** 9)
    pol = politica_de_genoma(g, config, preprocesador(cfg, version))
    print('Ejecutando {} (Ctrl-C para parar)'.format(ruta))
    try:
        obs, _ = env.reset(options={'reinicio': 'escena'})
        while True:
            obs, _, term, trunc, info = env.step(pol(obs))
            if term or trunc:
                print('Fin: {} a los {:.1f} s'.format(info['causa'], info['t']))
                obs, _ = env.reset(options={'reinicio': 'escena'})
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# =====================================================================

def main():
    p = argparse.ArgumentParser(description='Práctica 2 de RIA: NEAT sobre el entorno de la P1.')
    m = p.add_mutually_exclusive_group(required=True)
    for n in ['medir', 'manual', 'evolucionar', 'reanudar', 'reevaluar', 'ejecutar']:
        m.add_argument('--' + n, action='store_true')
    p.add_argument('--etiqueta', default='C1', help='configuración del guía: C1, C2, C3, C4')
    p.add_argument('--formulacion', choices=['A', 'B'], default='A',
                   help='A = producto, B = suma ponderada')
    p.add_argument('--semilla', type=int, default=1)
    p.add_argument('--generaciones', type=int, default=30, help='total de la evolución')
    p.add_argument('--pasos', type=int, default=100, help='pasos máximos por episodio (N)')
    p.add_argument('--episodios-ind', type=int, default=1, help='episodios por individuo')
    p.add_argument('--agregar', choices=['media', 'min'], default='media')
    p.add_argument('--reinicio', choices=['escena', 'mixto', 'colocar'], default='escena',
                   help='cómo empieza cada episodio: escena (como el protocolo; todos los '
                        'individuos en igualdad de condiciones) o mixto/colocar (el seguidor '
                        'aparece detrás del guía donde esté: más variedad pero mucho más ruido)')
    p.add_argument('--desde', default=None,
                   help='--evolucionar: carpeta de otra evolución cuya población final se '
                        'continúa (evolución incremental, p. ej. C1 -> C2)')
    p.add_argument('--multiplicador', default='1.0',
                   help='multiplicador de velocidad de RoboboSim (sólo se anota en meta.json)')
    p.add_argument('--episodios', type=int, default=3, help='--manual / --reevaluar: episodios')
    p.add_argument('--candidatos', type=int, default=5, help='--reevaluar: nº de genomas')
    p.add_argument('--config-ini', default='config-p2.ini')
    p.add_argument('--config-p1', default=os.path.join(P1_DIR, 'p1_config.json'))
    p.add_argument('--salida', default=None, help='carpeta de la evolución')
    p.add_argument('--sin-pausa', action='store_true')
    p.add_argument('--centinela', type=int, default=5,
                   help='--evolucionar/--reanudar: cada cuántas generaciones (y al empezar) se '
                        'comprueba con el controlador a mano que el simulador sigue sano; 0 = nunca')
    p.add_argument('--detalle', action='store_true', help='una línea por paso (diagnóstico)')
    args = p.parse_args()
    OPCIONES['detalle'] = args.detalle

    if args.medir:
        modo_medir(args)
    elif args.manual:
        modo_manual(args)
    elif args.evolucionar:
        modo_evolucionar(args)
    elif args.reanudar:
        modo_evolucionar(args, reanudar=True)
    elif args.reevaluar:
        modo_reevaluar(args)
    else:
        modo_ejecutar(args)


if __name__ == '__main__':
    main()
