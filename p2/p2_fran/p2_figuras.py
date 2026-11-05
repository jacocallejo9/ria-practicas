#!/usr/bin/env python3
"""
Práctica 2 — Figuras de la evolución (no usa ROS ni el simulador).

Lee evolucion.csv y especies.csv de una o varias evoluciones (semillas) y
genera, con código propio:

  <salida>_calidad.png/.pdf       calidad del mejor y media de la población
                                  por generación. Una semilla: banda = ±1
                                  desv. típica de la población. Varias: media
                                  de semillas y banda mínimo-máximo.
  <salida>_especies.png/.pdf      tamaño de cada especie (áreas apiladas)
  <salida>_complejidad.png/.pdf   nodos y conexiones del mejor, y % de
                                  episodios completados por generación

    python3 p2_figuras.py --run p2_runs/C1_A_s1 --run p2_runs/C1_A_s2 \\
        --referencia p2_runs/referencia_C1_A.json --titulo "C1 (10-10)" --salida figuras/p2_C1

El eje horizontal es el índice de generación (fila del CSV). El pie de figura
con población, pasos totales y horas se imprime por consola y se dibuja bajo
cada figura.
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def leer_run(d):
    with open(os.path.join(d, 'evolucion.csv')) as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise SystemExit('Sin generaciones en ' + d)
    r = {k: np.array([float(x[k]) for x in filas]) for k in filas[0]}
    r['nombre'] = os.path.basename(os.path.normpath(d))
    esp = os.path.join(d, 'especies.csv')
    r['tam'] = None
    if os.path.exists(esp):
        with open(esp) as f:
            datos = list(csv.DictReader(f))
        ids = sorted({int(x['especie']) for x in datos})
        n = len(filas)
        m = np.zeros((len(ids), n))
        for x in datos:
            i = int(x['indice'])
            if i < n:
                m[ids.index(int(x['especie'])), i] = float(x['tam'])
        r['tam'] = m
    r['pop'] = int(r['tam'][:, -1].sum()) if r['tam'] is not None else 0
    return r


def leer_referencia(v):
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        with open(v) as f:
            return float(json.load(f)['calidad_media'])


def pie(runs, args):
    r0 = runs[0]
    pasos = ', '.join('{:.0f}'.format(r['pasos'][-1]) for r in runs)
    horas = ', '.join('{:.1f}'.format(r['segundos'][-1] / 3600) for r in runs)
    return ('{} semilla(s); población {}; {} generaciones; {} ep./individuo; episodio máx. {} pasos. '
            'Pasos del entorno: {}. Horas: {}.').format(
                len(runs), r0['pop'], len(r0['cal_mejor']), args.episodios_ind, args.pasos_max,
                pasos, horas)


def guardar(fig, ruta, texto):
    fig.text(0.01, -0.01, texto, fontsize=7, va='top', wrap=True)
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig('{}.{}'.format(ruta, ext), dpi=150, bbox_inches='tight')
    print('Figura: {}.png y .pdf\n  Pie: {}'.format(ruta, texto))


def fig_calidad(runs, ref, args):
    n = min(len(r['cal_mejor']) for r in runs)
    g = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for clave, color, nombre in (('cal_mejor', 'tab:blue', 'Mejor individuo'),
                                 ('cal_media', 'tab:orange', 'Media de la población')):
        M = np.array([r[clave][:n] for r in runs])
        if len(runs) == 1:
            ax.plot(g, M[0], color=color, lw=2, label=nombre)
            if clave == 'cal_media':
                sd = runs[0]['cal_desv'][:n]
                ax.fill_between(g, np.clip(M[0] - sd, 0, 100), np.clip(M[0] + sd, 0, 100),
                                color=color, alpha=0.25, lw=0, label='±1 desv. típica de la población')
        else:
            for m in M:
                ax.plot(g, m, color=color, alpha=0.2, lw=0.7)
            ax.plot(g, M.mean(0), color=color, lw=2, label=nombre + ' (media de semillas)')
            ax.fill_between(g, M.min(0), M.max(0), color=color, alpha=0.2, lw=0,
                            label=nombre + ': mínimo-máximo')
    if ref is not None:
        ax.axhline(ref, color='k', ls='--', lw=1.2, label='controlador a mano ({:.1f})'.format(ref))
    ax.set_ylim(0, 100)
    ax.set_xlabel('Generación')
    ax.set_ylabel('Calidad F (adimensional, rango teórico 0-100)')
    ax.set_title('Evolución de la calidad · ' + args.titulo, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    guardar(fig, args.salida + '_calidad', pie(runs, args))


def fig_especies(runs, args):
    con = [r for r in runs if r['tam'] is not None]
    if not con:
        print('Sin especies.csv: no se dibuja la especiación.')
        return
    fig, ejes = plt.subplots(len(con), 1, figsize=(8, 2.8 * len(con) + 0.8), sharex=True, squeeze=False)
    for ax, r in zip(ejes[:, 0], con):
        ax.stackplot(np.arange(r['tam'].shape[1]), r['tam'])
        ax.set_ylabel('Individuos por especie')
        n_esp = (r['tam'] > 0).sum(axis=0)
        ax.set_title('{} · especies vivas: mín {} / máx {} / final {}'.format(
            r['nombre'], n_esp.min(), n_esp.max(), n_esp[-1]), fontsize=9, loc='left')
        ax.grid(alpha=0.3)
    ejes[-1, 0].set_xlabel('Generación')
    fig.tight_layout()
    guardar(fig, args.salida + '_especies', pie(runs, args))


def fig_complejidad(runs, args):
    n = min(len(r['cal_mejor']) for r in runs)
    g = np.arange(n)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for k, r in enumerate(runs):
        a.plot(g, r['nodos_mejor'][:n], color='tab:blue', alpha=0.4 + 0.6 / len(runs),
               label='nodos (sin entradas)' if k == 0 else None)
        a.plot(g, r['conexiones_mejor'][:n], color='tab:red', alpha=0.4 + 0.6 / len(runs),
               label='conexiones activas' if k == 0 else None)
    a.set_ylabel('Complejidad del mejor individuo')
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    M = np.array([r['pct_completados'][:n] for r in runs])
    b.plot(g, M.mean(0), color='tab:green', lw=2, label='episodios completados (media de semillas)')
    if len(runs) > 1:
        b.fill_between(g, M.min(0), M.max(0), color='tab:green', alpha=0.2, lw=0, label='mínimo-máximo')
    b.set_ylim(-2, 102)
    b.set_xlabel('Generación')
    b.set_ylabel('Episodios completados (%)')
    b.grid(alpha=0.3)
    b.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    guardar(fig, args.salida + '_complejidad', pie(runs, args))


def main():
    ap = argparse.ArgumentParser(description='Figuras de evolución de la práctica 2.')
    ap.add_argument('--run', action='append', required=True,
                    help='carpeta de una evolución (repetir una por semilla)')
    ap.add_argument('--referencia', default=None,
                    help='calidad del controlador a mano: número o JSON de p2_neat.py --manual')
    ap.add_argument('--titulo', default='')
    ap.add_argument('--salida', default='figuras/p2')
    ap.add_argument('--episodios-ind', type=int, default=1, help='sólo para el pie')
    ap.add_argument('--pasos-max', type=int, default=80, help='sólo para el pie')
    args = ap.parse_args()
    runs = [leer_run(d) for d in args.run]
    ref = leer_referencia(args.referencia)
    fig_calidad(runs, ref, args)
    fig_especies(runs, args)
    fig_complejidad(runs, args)
    for r in runs:
        print('{}: mejor F = {:.2f} (gen {}), media final = {:.2f}, completados final = {:.0f} %'.format(
            r['nombre'], r['cal_mejor'].max(), int(np.argmax(r['cal_mejor'])),
            r['cal_media'][-1], r['pct_completados'][-1]))


if __name__ == '__main__':
    main()
