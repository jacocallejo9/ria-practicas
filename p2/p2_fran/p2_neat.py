#!/usr/bin/env python3
"""
Práctica 2 — Evolución de un controlador con NEAT-Python sobre el entorno
de la Práctica 1 (RoboboSeguidorEnv).

Reutiliza de la P1: p1_seguidor_env.py (entorno), p1_comun.py,
p1_simulador.py y p1_config.json. Nada de eso se modifica.

--------------------------------------------------------------------
PUESTA EN MARCHA (igual que en la P1)
--------------------------------------------------------------------
  RoboboSim en Follow City, guía configurado y Apply (C1 = Mode 1, 10-10;
  C2 = Mode 1, 10-40; C3 = Mode 2, 10-40), ventana en primer plano.
  Puente lanzado sólo con el módulo de blobs. Luego, en la terminal de
  ejecución (cd /docker-ros/ws, source /root/robobo_ws/install/setup.bash):

  python3 p2_calidad.py                          # tabla de la calidad
  python3 p2_neat.py --medir                     # tiempos y presupuesto
  python3 p2_neat.py --evolucionar --etiqueta C1 --semilla 1 --generaciones 25
  python3 p2_neat.py --reanudar    --etiqueta C1 --semilla 1 --generaciones 25
  python3 p2_neat.py --reevaluar   --etiqueta C1 --semilla 1     # elige el genoma final
  python3 p2_neat.py --ejecutar    --etiqueta C1 --semilla 1     # vídeo
  python3 p2_figuras.py --run p2_runs/C1_A_s1 --run p2_runs/C1_A_s2 \\
          --titulo "C1" --salida figuras/p2_C1

--------------------------------------------------------------------
DECISIONES DE DISEÑO (para la memoria)
--------------------------------------------------------------------
  Observación: 6 de las 11 componentes de la P1 (OBS_IDX): visible, x,
    distancia, vel_distancia, vel_guia, ir_frontal. Se quitan y, sin_pelota,
    vel_x y las acciones previas: la red es feed-forward y pequeña, y NEAT
    se beneficia de pocas entradas. Ya están normalizadas en [-1, 1] o
    [0, 1,5] por el entorno.
  Acción: Box(-1, 1)^2 (v, w), salida tanh -> el entorno la reescala a
    velocidades reales exactamente como en la P1.
  Episodio de evaluación: N = 80 pasos (~45 s), siempre con reinicio de
    escena (todos los individuos parten de lo mismo y el guía ya tiene
    curva en C1 antes de que acabe el episodio). Modo 'entrenamiento' del
    entorno: además de los fallos del protocolo corta por salirse de la
    calzada y por adelantar al guía.
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
import random
import re
import shutil
import time

import numpy as np
import neat

_AQUI = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_AQUI, 'p1_comun.py')):      # P1 copiada junto a la P2
    from p1_comun import cargar_config
    P1_DIR = _AQUI
else:                                                       # P1 en otra carpeta (p2_rutas.py)
    from p2_rutas import P1_DIR, cargar_cfg_p1 as cargar_config
from p2_calidad import Calidad
from p2_registro import RegistroGeneracion, CONTADOR, ESTADO

OBS_IDX = [0, 1, 3, 5, 6, 8]
NOMBRES_ENTRADA = ['visible', 'x', 'distancia', 'vel_dist', 'vel_guia', 'ir_frontal']
NOMBRES_SALIDA = ['v', 'w']
GUIA = {'C1': 'Mode 1, 10-10 (crucero constante)', 'C2': 'Mode 1, 10-40 (crucero)',
        'C3': 'Mode 2, 10-40 (variable)'}


class ErrorInfraestructura(Exception):
    """Fallo del simulador o del puente: no es culpa del individuo."""


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


def crear_env(cfg, n_pasos):
    from p1_seguidor_env import RoboboSeguidorEnv
    return RoboboSeguidorEnv(config=cfg, modo='entrenamiento', pasos_max=n_pasos,
                             verbose=OPCIONES['detalle'])


def directorio_run(args):
    return args.salida or os.path.join('p2_runs', '{}_{}_s{}'.format(
        args.etiqueta, args.formulacion, args.semilla))


def cargar_neat(ruta_ini):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, ruta_ini)
    n_in = config.genome_config.num_inputs
    if n_in != len(OBS_IDX):
        raise SystemExit('num_inputs = {} en {} pero OBS_IDX tiene {} componentes.'.format(
            n_in, ruta_ini, len(OBS_IDX)))
    return config


def politica_de_genoma(genoma, config, idx=None):
    """Función observación completa (11) -> acción (2) a partir de un genoma."""
    idx = OBS_IDX if idx is None else idx
    red = neat.nn.FeedForwardNetwork.create(genoma, config)
    return lambda obs: np.clip(np.array(red.activate(np.asarray(obs)[idx]),
                                        dtype=np.float32), -1.0, 1.0)


# =====================================================================
# Evaluación de individuos
# =====================================================================

class Evaluador:
    """Ejecuta episodios del entorno y calcula la calidad (p2_calidad.py)."""

    CAMPOS_LOG = ['generacion', 'genoma', 'episodio', 'calidad', 'pasos', 'causa',
                  'completado', 'c_sep', 'c_ori', 'c_car', 'segundos', 'reinicio']

    def __init__(self, env, calidad, episodios=1, agregar='media', ruta_log=None, idx=None,
                 reinicio='escena', prob_escena=0.3, semilla=0):
        self.env, self.calidad = env, calidad
        self.reinicio, self.prob_escena = reinicio, prob_escena
        self.rng = random.Random(semilla)        # propio: no toca el generador de NEAT
        self.N = calidad.N
        self.k, self.agregar = episodios, agregar
        self.ruta_log = ruta_log
        self.idx = OBS_IDX if idx is None else idx
        if ruta_log and not os.path.exists(ruta_log):
            os.makedirs(os.path.dirname(ruta_log) or '.', exist_ok=True)
            with open(ruta_log, 'w', newline='') as f:
                csv.writer(f).writerow(self.CAMPOS_LOG)

    def tipo_reinicio(self):
        """'escena' (como el protocolo) o 'colocar' (el seguidor aparece detrás del
        guía en un punto cualquiera del recorrido: rectas y curvas)."""
        if self.reinicio == 'mixto':
            return 'escena' if self.rng.random() < self.prob_escena else 'colocar'
        return self.reinicio

    def episodio(self, politica):
        """Un episodio completo -> dict. Repite si el simulador se congela."""
        for _ in range(4):
            t0 = time.time()
            try:
                tipo = self.tipo_reinicio()
                obs, info = self.env.reset(options={'reinicio': tipo})
                suma, vivos, causa = 0.0, 0, ''
                comp = np.zeros(3)
                for _ in range(self.N):
                    obs, _, term, trunc, info = self.env.step(politica(obs))
                    vivos += 1
                    causa = info['causa']
                    if causa == 'congelado':
                        break
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
            completado = causa == 'tiempo'
            CONTADOR['pasos'] += vivos
            CONTADOR['episodios'] += 1
            CONTADOR['completados'] += int(completado)
            return {'calidad': self.calidad.total(suma, completado), 'pasos': vivos,
                    'causa': causa or 'tiempo', 'completado': completado,
                    'c_sep': comp[0] / vivos, 'c_ori': comp[1] / vivos, 'c_car': comp[2] / vivos,
                    'segundos': time.time() - t0, 'reinicio': tipo}
        raise ErrorInfraestructura('El simulador sigue congelado tras 4 intentos.')

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


def crear_eval_genomes(ev, config_neat):
    def eval_genomes(genomas, config):
        for n, (gid, g) in enumerate(genomas, 1):
            valor, res = ev.evaluar(politica_de_genoma(g, config, ev.idx), gid)
            g.fitness = valor
            CONTADOR['evaluaciones'] += 1
            r = res[-1]
            print('  gen {:3d}  ind {:2d}/{:2d} (id {:4d})  F = {:6.2f}  {:3d} pasos  {:13s} {:8s} '
                  'sep {:.2f} ori {:.2f} car {:.2f}'.format(
                      ESTADO['generacion'], n, len(genomas), gid, valor, r['pasos'], r['causa'],
                      '(' + r['reinicio'] + ')', r['c_sep'], r['c_ori'], r['c_car']))
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


def modo_evolucionar(args, reanudar=False):
    run = directorio_run(args)
    os.makedirs(os.path.join(run, 'chk'), exist_ok=True)
    ini = os.path.join(run, 'config-p2.ini')
    if not os.path.exists(ini):
        preparar_config(args.config_ini, ini, args.semilla)
    config = cargar_neat(ini)
    cfg = cargar_config(args.config_p1)
    asegurar_linea(cfg)
    shutil.copy(args.config_p1, os.path.join(run, 'p1_config.json'))
    meta = {'etiqueta': args.etiqueta, 'formulacion': args.formulacion, 'semilla': args.semilla,
            'n_pasos': args.pasos, 'episodios_por_individuo': args.episodios_ind,
            'agregar': args.agregar, 'reinicio': args.reinicio, 'obs_idx': OBS_IDX, 'entradas': NOMBRES_ENTRADA,
            'salidas': NOMBRES_SALIDA, 'guia': GUIA.get(args.etiqueta, '')}
    with open(os.path.join(run, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if not args.sin_pausa:
        input('\nConfigura el guía en RoboboSim como {} -> {}, pulsa Apply, deja la ventana '
              'en primer plano y pulsa Enter... '.format(args.etiqueta, GUIA.get(args.etiqueta, '?')))

    random.seed(args.semilla)
    ya = 0
    if reanudar:
        chk = sorted(glob.glob(os.path.join(run, 'chk', 'p2-*')),
                     key=lambda r: int(r.rsplit('-', 1)[1]))
        if not chk:
            raise SystemExit('No hay puntos de control en {}/chk'.format(run))
        poblacion = neat.Checkpointer.restore_checkpoint(chk[-1])
        ruta_csv = os.path.join(run, 'evolucion.csv')
        if os.path.exists(ruta_csv):
            with open(ruta_csv) as f:
                ya = max(sum(1 for _ in f) - 1, 0)
        print('Reanudando desde {} ({} generaciones ya hechas).'.format(chk[-1], ya))
    else:
        if os.path.exists(os.path.join(run, 'evolucion.csv')):
            raise SystemExit('{} ya tiene una evolución; usa --reanudar o otra --salida.'.format(run))
        poblacion = neat.Population(config)
    faltan = args.generaciones - ya
    if faltan <= 0:
        raise SystemExit('Ya están hechas las {} generaciones.'.format(args.generaciones))

    env = crear_env(cfg, args.pasos)
    ev = Evaluador(env, Calidad(cfg, args.formulacion, args.pasos), args.episodios_ind,
                   args.agregar, os.path.join(run, 'evaluaciones.csv'),
                   reinicio=args.reinicio, prob_escena=cfg['reinicio']['prob_escena'],
                   semilla=args.semilla)
    poblacion.add_reporter(neat.StdOutReporter(True))
    estadisticas = neat.StatisticsReporter()
    poblacion.add_reporter(estadisticas)
    poblacion.add_reporter(neat.Checkpointer(generation_interval=1,
                                             filename_prefix=os.path.join(run, 'chk', 'p2-')))
    poblacion.add_reporter(RegistroGeneracion(os.path.join(run, 'evolucion.csv'),
                                              os.path.join(run, 'genomas')))
    print('\nEvolución {} · formulación {} · semilla {} · {} generaciones de {} individuos · '
          '{} pasos/episodio -> {}'.format(args.etiqueta, args.formulacion, args.semilla, faltan,
                                          config.pop_size, args.pasos, run))
    try:
        mejor = poblacion.run(crear_eval_genomes(ev, config), faltan)
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
        g = next(iter(neat.Population(config).population.values()))   # genoma aleatorio inicial
        ev = Evaluador(env, Calidad(cfg, args.formulacion, args.pasos), 1, 'media', None)
        valor, res = ev.evaluar(politica_de_genoma(g, config), registrar=False)
        r = res[0]
        t_paso = (r['segundos'] - t_reset[-1]) / max(r['pasos'], 1)
        print('Genoma aleatorio: F = {:.2f}, {} pasos, {}, ~{:.2f} s/paso'.format(
            valor, r['pasos'], r['causa'], t_paso))
        T = args.pasos * (t_paso if r['pasos'] > 5 else 0.55)
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
    config = cargar_neat(os.path.join(run, 'config-p2.ini'))
    with open(os.path.join(run, 'meta.json')) as f:
        meta = json.load(f)
    cfg = cargar_config(os.path.join(run, 'p1_config.json'))
    asegurar_linea(cfg)
    with open(os.path.join(run, 'evolucion.csv')) as f:
        filas = list(csv.DictReader(f))
    orden = sorted(filas, key=lambda r: -float(r['cal_mejor']))
    candidatos, vistos = [], set()
    rutas = [os.path.join(run, 'genomas', 'mejor_gen{:03d}.pkl'.format(int(r['generacion'])))
             for r in orden]
    extra = os.path.join(run, 'mejor_genoma.pkl')
    if os.path.exists(extra):
        rutas.insert(0, extra)
    for ruta in rutas:
        if not os.path.exists(ruta):
            continue
        with open(ruta, 'rb') as f:
            g = pickle.load(f)
        if g.key in vistos:
            continue
        vistos.add(g.key)
        candidatos.append((ruta, g))
        if len(candidatos) >= args.candidatos:
            break
    print('Reevaluando {} candidatos x {} episodios (formulación {})'.format(
        len(candidatos), args.episodios, meta['formulacion']))
    env = crear_env(cfg, meta['n_pasos'])
    try:
        if not args.sin_pausa:
            input('Guía como {} ({}), Apply y Enter... '.format(
                meta['etiqueta'], GUIA.get(meta['etiqueta'], '?')))
        ev = Evaluador(env, Calidad(cfg, meta['formulacion'], meta['n_pasos']), 1, 'media',
                       None, meta['obs_idx'])
        tabla = []
        for ruta, g in candidatos:
            valor, res = ev.evaluar(politica_de_genoma(g, config, meta['obs_idx']), g.key,
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
    config = cargar_neat(os.path.join(run, 'config-p2.ini'))
    with open(os.path.join(run, 'meta.json')) as f:
        meta = json.load(f)
    ruta = os.path.join(run, 'genoma_final.pkl')
    if not os.path.exists(ruta):
        ruta = os.path.join(run, 'mejor_genoma.pkl')
    with open(ruta, 'rb') as f:
        g = pickle.load(f)
    cfg = cargar_config(os.path.join(run, 'p1_config.json'))
    asegurar_linea(cfg)
    env = crear_env(cfg, 10 ** 9)
    pol = politica_de_genoma(g, config, meta['obs_idx'])
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
    for n in ['medir', 'evolucionar', 'reanudar', 'reevaluar', 'ejecutar']:
        m.add_argument('--' + n, action='store_true')
    p.add_argument('--etiqueta', default='C1', help='configuración del guía: C1, C2, C3')
    p.add_argument('--formulacion', choices=['A', 'B'], default='A',
                   help='A = producto, B = suma ponderada')
    p.add_argument('--semilla', type=int, default=1)
    p.add_argument('--generaciones', type=int, default=25, help='total de la evolución')
    p.add_argument('--pasos', type=int, default=80, help='pasos máximos por episodio (N)')
    p.add_argument('--episodios-ind', type=int, default=1, help='episodios por individuo')
    p.add_argument('--agregar', choices=['media', 'min'], default='media')
    p.add_argument('--reinicio', choices=['escena', 'mixto'], default='mixto',
                   help='cómo empieza cada episodio de la evolución: escena (siempre desde el '
                        'inicio) o mixto (escena con prob. reinicio.prob_escena de p1_config.json '
                        'y, si no, el seguidor aparece en un punto cualquiera del recorrido)')
    p.add_argument('--episodios', type=int, default=3, help='--reevaluar: episodios por candidato')
    p.add_argument('--candidatos', type=int, default=5, help='--reevaluar: nº de genomas')
    p.add_argument('--config-ini', default='config-p2.ini')
    p.add_argument('--config-p1', default=os.path.join(P1_DIR, 'p1_config.json'))
    p.add_argument('--salida', default=None, help='carpeta de la evolución')
    p.add_argument('--sin-pausa', action='store_true')
    p.add_argument('--detalle', action='store_true', help='una línea por paso (diagnóstico)')
    args = p.parse_args()
    OPCIONES['detalle'] = args.detalle

    if args.medir:
        modo_medir(args)
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