#!/usr/bin/env python3
"""
Práctica 2 — Protocolo de evaluación del anexo B (C1-C4).

Es el evaluador de la P1 (p1_protocolo.py, copiado SIN MODIFICAR) ampliado
como pide el enunciado (apartado 6.4 y anexo B):

  * una configuración más, C4 (variable con paradas), que se ejecuta dos
    veces por el carácter aleatorio de las paradas (B.2);
  * la métrica de paradas (B.4, sólo C4): número de paradas del guía y
    número de ellas superadas.

El formato de los CSV (una fila por paso) es el de la P1, así que las
métricas de C1-C3 se calculan con exactamente el mismo código que en la P1
y los resultados de las dos prácticas son comparables.

--------------------------------------------------------------------
USO (terminal 2, con el puente en la terminal 1 sólo con 'blob')
--------------------------------------------------------------------
  Configura el guía en RoboboSim, pulsa Apply y deja la ventana en primer
  plano. Cada orden evalúa una configuración (C4 hace 2 episodios):

  python3 p2_protocolo.py --evaluar --run p2_runs/C1_A_s1 --configuracion C1
  python3 p2_protocolo.py --evaluar --manual              --configuracion C4
  python3 p2_protocolo.py --evaluar --aleatoria           --configuracion C2
  # Línea base de la P1 (misma sesión, simulador y multiplicador, apartado 6.5):
  python3 p2_protocolo.py --evaluar --modelo-p1 ../../p1/P1_definitivo/p1_logs/sac_C1_s0/modelo \\
          --configuracion C1

  Tablas (no necesitan ROS, se pueden rehacer fuera del contenedor):
  python3 p2_protocolo.py p2_eval/*.csv
  python3 p2_protocolo.py p2_eval/*.csv --markdown > tabla.md
  python3 p2_protocolo.py p2_eval/*C4*.csv --paradas      # detalle de cada parada

--------------------------------------------------------------------
MÉTRICA DE PARADAS (cómo se calcula, para la memoria)
--------------------------------------------------------------------
  Parada del guía (B.1): intervalo en el que la velocidad del guía,
    calculada a partir de su pose (columnas guia_x, guia_z del CSV), es
    menor que 0,01 m/s durante más de 0,5 s. La velocidad se calcula entre
    pasos consecutivos (~0,6 s). No cuenta la espera inicial: tras
    reiniciar la escena el guía está quieto a propósito hasta que el
    seguidor se le acerca, y eso no es una parada del modo con paradas.
  Parada superada (B.4): el seguidor se detiene sin colisión y sigue sin
    fallo 10 s después de que el guía reanude la marcha.
      - "Se detiene": su velocidad (de su pose, seg_x, seg_z) baja de
        0,01 m/s en algún tramo entre pasos dentro de la parada del guía
        (mismo umbral que para el guía).
      - "Sin colisión y sin fallo": el episodio no termina por un fallo
        de B.3 (ni por otra terminación propia) antes de 10 s después del
        fin de la parada.
  Una parada queda SIN EVALUAR (se informa aparte y no cuenta como
    superada ni como no superada) si el episodio acaba sin fallo antes de
    poder comprobarla: a los 120 s con el guía aún parado o con menos de
    10 s desde que reanudó, o por un episodio interrumpido o congelado.
"""

import argparse
import glob
import os
import time
from collections import Counter

import numpy as np

from p1_protocolo import (RegistroEvaluacion, leer_episodios, metricas, agrupar,
                          _texto_desenlaces)

CONFIGURACIONES = ['C1', 'C2', 'C3', 'C4']
# Guía en RoboboSim (anexo B.2) y número de ejecuciones de cada configuración
GUIA = {'C1': ('Crucero, velocidad 10-10', 1),
        'C2': ('Crucero, velocidad 10-40', 1),
        'C3': ('Variable, velocidad 10-40', 1),
        'C4': ('Variable con paradas, velocidad 10-40 (probabilidad y tiempos '
               'de parada los del enunciado)', 2)}

UMBRAL_V = 0.01          # m/s: guía (B.1) y seguidor parados
DURACION_PARADA_S = 0.5  # una parada dura más de esto (B.1)
MARGEN_S = 10.0          # sin fallo 10 s después de reanudar (B.4)
RUTA_EVAL = 'p2_eval'

# Desenlaces que no son culpa del controlador: no permiten juzgar la parada
NO_EVALUABLE = ('interrumpido', 'congelado')


# =====================================================================
# Métrica de paradas
# =====================================================================

def _tramos(filas, cx, cz):
    """(t, v): instantes de las filas válidas y velocidad entre filas
    consecutivas (m/s) a partir de la pose."""
    t, p = [], []
    for f in filas:
        if f['t_s'] != '' and f[cx] != '' and f[cz] != '':
            t.append(float(f['t_s']))
            p.append((float(f[cx]), float(f[cz])))
    t, p = np.array(t), np.array(p).reshape(-1, 2)
    if len(t) < 2:
        return t, np.array([])
    dt = np.maximum(np.diff(t), 1e-6)
    return t, np.linalg.norm(np.diff(p, axis=0), axis=1) / dt


def paradas_guia(filas):
    """Paradas del guía de un episodio: lista de dicts con inicio y fin (s),
    duración y si reanuda la marcha antes de acabar el episodio."""
    t, v = _tramos(filas, 'guia_x', 'guia_z')
    if not v.size:
        return []
    quieto = v < UMBRAL_V
    movido = np.flatnonzero(~quieto)
    if not movido.size:
        return []                    # el guía no llegó a arrancar
    paradas = []
    k = movido[0]                    # se ignora la espera inicial tras el reinicio
    while k < len(v):
        if not quieto[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(v) and quieto[j + 1]:
            j += 1
        ini, fin = t[k], t[j + 1]
        if fin - ini > DURACION_PARADA_S:
            paradas.append({'inicio_s': float(ini), 'fin_s': float(fin),
                            'duracion_s': float(fin - ini),
                            'reanuda': j + 1 < len(v)})
        k = j + 1
    return paradas


def evaluar_paradas(episodio):
    """Paradas del guía en un episodio, cada una con su resultado:
    'superada', 'no_superada' (con motivo) o 'sin_evaluar'."""
    filas = episodio['filas']
    paradas = paradas_guia(filas)
    if not paradas:
        return []
    t_seg, v_seg = _tramos(filas, 'seg_x', 'seg_z')
    validos = [float(f['t_s']) for f in filas if f['t_s'] != '']
    t_fin = validos[-1] if validos else 0.0
    desenlace = episodio['desenlace']
    fallo = desenlace not in ('completado',) + NO_EVALUABLE
    for p in paradas:
        dentro = (t_seg[:-1] >= p['inicio_s'] - 1e-6) & (t_seg[1:] <= p['fin_s'] + 1e-6) \
            if v_seg.size else np.array([], dtype=bool)
        p['seguidor_v_min'] = float(v_seg[dentro].min()) if dentro.any() else float('nan')
        p['se_detiene'] = bool(dentro.any() and v_seg[dentro].min() < UMBRAL_V)
        if fallo and (not p['reanuda'] or t_fin <= p['fin_s'] + MARGEN_S):
            p['resultado'], p['motivo'] = 'no_superada', desenlace
        elif not p['reanuda']:
            p['resultado'], p['motivo'] = 'sin_evaluar', 'acaba con el guía parado'
        elif not p['se_detiene']:
            p['resultado'], p['motivo'] = 'no_superada', 'no se detiene'
        elif t_fin < p['fin_s'] + MARGEN_S:
            p['resultado'], p['motivo'] = 'sin_evaluar', \
                'acaba ({}) {:.1f} s después de reanudar'.format(desenlace, t_fin - p['fin_s'])
        else:
            p['resultado'], p['motivo'] = 'superada', ''
    return paradas


def metricas_p2(episodios):
    """Métricas de la P1 (B.4) más la de paradas, que sólo se calcula sobre
    los episodios de C4."""
    m = metricas(episodios)
    c4 = [e for e in episodios if e['configuracion'] == 'C4']
    res = Counter(p['resultado'] for e in c4 for p in evaluar_paradas(e))
    m.update(episodios_c4=len(c4), paradas=sum(res.values()),
             paradas_superadas=res.get('superada', 0),
             paradas_no_superadas=res.get('no_superada', 0),
             paradas_sin_evaluar=res.get('sin_evaluar', 0))
    return m


# =====================================================================
# Tablas
# =====================================================================

def _texto_paradas(m):
    if not m['episodios_c4']:
        return ''
    txt = '{}/{}'.format(m['paradas_superadas'], m['paradas'])
    if m['paradas_sin_evaluar']:
        txt += ' ({} sin evaluar)'.format(m['paradas_sin_evaluar'])
    return txt


def imprimir_tabla(grupos, markdown=False):
    """La tabla de la P1 con una columna más: paradas superadas/total (C4)."""
    cab = ['Política', 'Conf.', 'Completados', 'Desenlaces', 'En banda',
           'Sep. media', '|Desv.| media', 'Desv. < 15°', 'En carril', 'Paradas (C4)']
    filas = []
    for (pol, conf), eps in grupos.items():
        m = metricas_p2(eps)
        filas.append([
            pol, conf,
            '{}/{} ({:.0f} %)'.format(m['completados'], m['episodios'], m['pct_completados']),
            _texto_desenlaces(m['desenlaces']),
            '{:.1f} %'.format(m['pct_en_banda']),
            '{:.3f} m'.format(m['separacion_media']),
            '{:.1f}°'.format(m['desviacion_media']),
            '{:.1f} %'.format(m['pct_desviacion_ok']),
            '' if np.isnan(m['pct_en_carril']) else '{:.1f} %'.format(m['pct_en_carril']),
            _texto_paradas(m),
        ])
    if markdown:
        print('| ' + ' | '.join(cab) + ' |')
        print('|' + '---|' * len(cab))
        for f in filas:
            print('| ' + ' | '.join(f) + ' |')
        return
    anchos = [max(len(str(x)) for x in col) for col in zip(cab, *filas)]
    linea = '  '.join('{:<' + str(a) + '}' for a in anchos)
    print(linea.format(*cab))
    print('  '.join('-' * a for a in anchos))
    for f in filas:
        print(linea.format(*f))


def imprimir_paradas(episodios):
    """Una línea por parada del guía detectada (en cualquier configuración:
    en C1-C3 no debería haber ninguna; si la hay, revisar el umbral)."""
    print('\nParadas del guía')
    hay = False
    for e in episodios:
        for p in evaluar_paradas(e):
            hay = True
            print('  {} {} ep{}: {:6.1f}-{:6.1f} s ({:4.1f} s)  seguidor v_min {:.3f} m/s  '
                  '-> {}{}'.format(e['politica'], e['configuracion'], e['episodio'],
                                   p['inicio_s'], p['fin_s'], p['duracion_s'],
                                   p['seguidor_v_min'], p['resultado'],
                                   ' ({})'.format(p['motivo']) if p['motivo'] else ''))
            if e['configuracion'] != 'C4':
                print('    AVISO: parada fuera de C4')
    if not hay:
        print('  (ninguna)')


# =====================================================================
# --evaluar : ejecutar el protocolo en RoboboSim
# =====================================================================

def politica_neat(args):
    import pickle
    import p2_neat as P
    from p1_comun import cargar_config
    run = args.run.rstrip('/\\')
    meta = P.leer_meta(run)
    version = P.version_de_meta(meta)
    config = P.cargar_neat(os.path.join(run, 'config-p2.ini'), len(P.ENTRADAS[version]))
    ruta = args.genoma or os.path.join(run, 'genoma_final.pkl')
    if not os.path.exists(ruta):
        ruta = os.path.join(run, 'mejor_genoma.pkl')
        print('AVISO: no hay genoma_final.pkl (falta --reevaluar en {}); uso {}'.format(run, ruta))
    with open(ruta, 'rb') as f:
        g = pickle.load(f)
    cfg = cargar_config(os.path.join(run, 'p1_config.json'))
    nombre = 'neat_' + os.path.basename(run)
    if args.genoma:
        nombre += '_' + os.path.splitext(os.path.basename(args.genoma))[0]
    print('Genoma: {} ({} nodos, {} conexiones activas)'.format(
        ruta, len(g.nodes), sum(c.enabled for c in g.connections.values())))
    return P.politica_de_genoma(g, config, P.preprocesador(cfg, version)), nombre, cfg


def politica_p1(args):
    """Política de Stable-Baselines3 de la P1, con la configuración con la
    que se entrenó (config.json junto al modelo), como hace p1_seguidor.py."""
    from p1_comun import cargar_config
    from stable_baselines3 import PPO, SAC
    ruta = args.modelo_p1
    d = os.path.dirname(ruta.replace('.zip', ''))
    ruta_cfg = args.config_p1
    for r in (os.path.join(d, 'config.json'), os.path.join(os.path.dirname(d), 'config.json')):
        if os.path.exists(r):
            ruta_cfg = r
            break
    print('Configuración del modelo: ' + ruta_cfg)
    base = os.path.basename(ruta.replace('.zip', ''))
    nombre = (os.path.basename(d) if base == 'modelo' else base) or base
    Algoritmo = PPO if 'ppo' in ruta.lower() else SAC
    modelo = Algoritmo.load(ruta, device='cpu')
    cfg = cargar_config(ruta_cfg)
    # El signo de la x del blob depende de la sesión del simulador, no del
    # modelo: se toma el de la calibración actual (p1_config.json de la P2),
    # para que la política reciba la x con la convención con que se entrenó.
    actual = cargar_config(args.config_p1)['camara']['x_semiancho']
    if cfg['camara']['x_semiancho'] != actual:
        print('x_semiancho del modelo {} -> {} (calibración actual)'.format(
            cfg['camara']['x_semiancho'], actual))
        cfg['camara']['x_semiancho'] = actual
    return (lambda obs: modelo.predict(obs, deterministic=True)[0]), nombre, cfg


def elegir_politica(args):
    from p1_comun import cargar_config
    if args.run:
        return politica_neat(args)
    if args.modelo_p1:
        return politica_p1(args)
    cfg = cargar_config(args.config_p1)
    if args.manual:
        import p2_neat as P
        return P.controlador_manual(cfg), 'manual', cfg
    rng = np.random.default_rng(args.semilla)
    return (lambda obs: rng.uniform(-1.0, 1.0, 2).astype(np.float32)), 'aleatoria', cfg


def modo_evaluar(args):
    """Protocolo del anexo B, igual que p1_seguidor.py --evaluar.

    Cada episodio empieza con un reinicio de la escena y el seguidor
    parado, seguido de la acción de arranque predeterminada del entorno
    (B.2 la admite). Termina a los 120 s (completado) o en el primer paso
    con un fallo de B.3, calculado con la verdad del terreno. El entorno se
    usa en modo 'protocolo': sin las terminaciones propias del
    entrenamiento (salirse de la calzada, adelantar). La configuración del
    guía se fija a mano en RoboboSim; aquí sólo se anota.
    """
    from p1_comun import fallo_protocolo
    from p1_seguidor_env import RoboboSeguidorEnv
    import p2_neat as P
    politica, nombre, cfg = elegir_politica(args)
    P.asegurar_linea(cfg)
    p = cfg['protocolo']
    conf = args.configuracion
    n_ep = args.episodios or GUIA.get(conf, ('', 1))[1]
    env = RoboboSeguidorEnv(config=cfg, modo='protocolo', pasos_max=10 ** 9,
                            verbose=args.detalle)
    if not args.sin_pausa:
        input('\nConfigura el guía en RoboboSim como {} ({}), pulsa Apply, vuelve aquí, '
              'pulsa Enter y deja la ventana del simulador en primer plano... '.format(
                  conf, GUIA.get(conf, ('?',))[0]))
    fecha = time.strftime('%Y%m%d-%H%M%S')
    rutas = []
    try:
        for ep in range(1, n_ep + 1):
            ruta = os.path.join(RUTA_EVAL, '{}_{}_{}_ep{}.csv'.format(nombre, conf, fecha, ep))
            reg = RegistroEvaluacion(ruta)
            rutas.append(ruta)
            obs, info = env.reset(options={'reinicio': 'escena'})
            paso, causa, t = 0, '', 0.0
            try:
                while not causa:
                    obs, _, term, trunc, info = env.step(politica(obs))
                    paso += 1
                    t = info['t']
                    causa = fallo_protocolo(info['separacion'], info['t_sin_pelota'], p)
                    if not causa and (term or trunc):
                        causa = info['causa'] or 'otra'      # terminación propia o congelado
                    if not causa and t >= p['duracion_s']:
                        causa = 'completado'
                    reg.fila(configuracion=conf, politica=nombre, episodio=ep, paso=paso, t_s=t,
                             seg_x=info['pos_seguidor'][0], seg_z=info['pos_seguidor'][1],
                             pelota_x=info['pos_pelota'][0], pelota_z=info['pos_pelota'][1],
                             guia_x=info['pos_guia'][0], guia_z=info['pos_guia'][1],
                             separacion_m=info['separacion'], desviacion_deg=info['desviacion_deg'],
                             error_lateral_m=info['error_lateral'], en_banda=info['en_banda'],
                             desviacion_ok=info['desviacion_ok'], en_carril=info['en_carril'],
                             visible=info['visible'], x_blob=info['x'],
                             v_ms=info['v'], w_rads=info['w'], causa=causa)
            except KeyboardInterrupt:
                reg.fila(configuracion=conf, politica=nombre, episodio=ep, paso=paso + 1,
                         causa='interrumpido')
                raise
            finally:
                env._detener()
                reg.cerrar()
            print('episodio {}: {} a los {:.1f} s ({} pasos) -> {}'.format(ep, causa, t, paso, ruta))
            if causa == 'congelado':
                print('  El simulador se congeló: este episodio no vale, repítelo.')
    except KeyboardInterrupt:
        print('\nEvaluación interrumpida.')
    finally:
        env.close()
    if rutas:
        episodios = leer_episodios(rutas)
        print()
        imprimir_tabla(agrupar(episodios))
        if conf == 'C4':
            imprimir_paradas(episodios)


# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Protocolo del anexo B (P2): C1-C4 y métrica de paradas.')
    ap.add_argument('csv', nargs='*', help='CSV de evaluación (p2_eval/*.csv) para las tablas')
    ap.add_argument('--markdown', action='store_true')
    ap.add_argument('--paradas', action='store_true', help='detalle de cada parada del guía')
    ap.add_argument('--evaluar', action='store_true', help='ejecutar el protocolo en RoboboSim')
    pol = ap.add_mutually_exclusive_group()
    pol.add_argument('--run', default=None, help='carpeta de una evolución (p2_runs/C1_A_s1)')
    pol.add_argument('--modelo-p1', default=None, help='modelo de SB3 de la P1 (línea base)')
    pol.add_argument('--manual', action='store_true', help='controlador programado a mano')
    pol.add_argument('--aleatoria', action='store_true', help='política aleatoria')
    ap.add_argument('--genoma', default=None, help='con --run: otro .pkl en lugar de genoma_final')
    ap.add_argument('--configuracion', default='C1', choices=CONFIGURACIONES + ['otra'])
    ap.add_argument('--episodios', type=int, default=None,
                    help='por defecto los de B.2: 1 (C1-C3), 2 (C4)')
    ap.add_argument('--semilla', type=int, default=0, help='--aleatoria')
    ap.add_argument('--config-p1', default='p1_config.json')
    ap.add_argument('--sin-pausa', action='store_true')
    ap.add_argument('--detalle', action='store_true')
    args = ap.parse_args()

    if args.evaluar:
        if not (args.run or args.modelo_p1 or args.manual or args.aleatoria):
            ap.error('--evaluar necesita --run, --modelo-p1, --manual o --aleatoria')
        modo_evaluar(args)
        return
    if not args.csv:
        ap.error('indica los CSV (p2_eval/*.csv) o usa --evaluar')
    rutas = sorted({r for patron in args.csv for r in (glob.glob(patron) or [patron])})
    episodios = leer_episodios(rutas)
    if not episodios:
        raise SystemExit('No hay episodios en ' + ', '.join(args.csv))
    imprimir_tabla(agrupar(episodios), markdown=args.markdown)
    if args.paradas:
        imprimir_paradas(episodios)


if __name__ == '__main__':
    main()
