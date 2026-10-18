#!/usr/bin/env python3
"""
Práctica 1 — Curvas de aprendizaje a partir de los monitor.csv.

Cumple los requisitos del apartado "Curvas de aprendizaje" del enunciado:

  - Magnitudes: recompensa por episodio, duración del episodio (pasos) y
    fracción de episodios que terminan por cada causa.
  - Eje horizontal: pasos del entorno (suma acumulada de las duraciones
    de los episodios), nunca episodios ni tiempo de reloj.
  - Suavizado: media móvil de 'ventana' episodios y, detrás y tenue, la
    serie sin suavizar de cada semilla.
  - Semillas: media de las semillas y banda entre el mínimo y el máximo.
  - Referencia: recompensa media por episodio de la política aleatoria en
    el mismo entorno (p1_seguidor.py --aleatorio).

Cada grupo (por ejemplo SAC y PPO) se da como nombre=patrón; el patrón
apunta a los directorios de las semillas (o a sus monitor.csv):

    python3 p1_curvas.py --grupo "SAC=p1_logs/sac_C3_s*" \\
                         --grupo "PPO=p1_logs/ppo_C3_s*" \\
                         --aleatoria p1_logs/aleatoria_C3 \\
                         --ventana 10 --titulo "Guía C3 (variable 10-40)" \\
                         --salida figuras/curvas_C3

No usa ROS 2. Si el matplotlib del contenedor falla por la versión de
NumPy, se puede ejecutar en Windows (python p1_curvas.py ...) porque sólo
lee los CSV de esta carpeta, o arreglarlo en el contenedor con
    pip install --user --upgrade matplotlib
"""

import argparse
import csv
import glob
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COLORES = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
COLORES_CAUSA = {'tiempo': 'tab:green', 'colision': 'tab:red', 'guia_perdido': 'tab:orange',
                 'sin_pelota': 'tab:purple', 'fuera_calzada': 'tab:brown',
                 'fuera_carril': 'tab:pink', 'adelanta': 'tab:gray'}


def leer_monitor(ruta):
    """monitor.csv de Stable-Baselines3 -> pasos acumulados, r, l y causas."""
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, 'monitor.csv')
    with open(ruta, newline='') as f:
        primera = f.readline()
        if not primera.startswith('#'):          # sin la cabecera JSON
            f.seek(0)
        filas = list(csv.DictReader(f))
    r = np.array([float(x['r']) for x in filas])
    l = np.array([int(float(x['l'])) for x in filas])
    causas = [x.get('causa', '') or 'tiempo' for x in filas]
    return {'ruta': ruta, 'pasos': np.cumsum(l), 'r': r, 'l': l, 'causas': causas}


def media_movil(y, ventana):
    """Media de los últimos 'ventana' episodios (ventana hacia atrás)."""
    c = np.cumsum(np.insert(np.asarray(y, dtype=float), 0, 0.0))
    i = np.arange(1, len(y) + 1)
    ini = np.maximum(i - ventana, 0)
    return (c[i] - c[ini]) / (i - ini)


def en_rejilla(pasos_ep, y, rejilla):
    """Valor del último episodio terminado en cada punto de la rejilla."""
    idx = np.searchsorted(pasos_ep, rejilla, side='right') - 1
    out = np.full(len(rejilla), np.nan)
    ok = idx >= 0
    out[ok] = np.asarray(y)[idx[ok]]
    return out


def expandir(patron):
    rutas = sorted(glob.glob(patron)) or [patron]
    return [r for r in rutas if os.path.exists(r if not os.path.isdir(r)
                                                else os.path.join(r, 'monitor.csv'))]


def rejilla_grupo(runs):
    """Pasos donde se promedian las semillas de un grupo: desde que todas
    han terminado su primer episodio hasta donde llegan TODAS (así la banda
    no mezcla tramos con distinto número de semillas). Cada grupo tiene la
    suya, para no cortar a uno más largo que otro."""
    return np.linspace(max(r['pasos'][0] for r in runs), min(r['pasos'][-1] for r in runs), 300)


def panel(ax, grupos, campo, ventana, etiqueta_y, referencia=None):
    for k, (nombre, runs) in enumerate(grupos):
        color = COLORES[k % len(COLORES)]
        rejilla = rejilla_grupo(runs)
        suav = []
        for run in runs:
            y = run[campo]
            ax.plot(run['pasos'], y, color=color, alpha=0.15, lw=0.7)
            suav.append(en_rejilla(run['pasos'], media_movil(y, ventana), rejilla))
        suav = np.array(suav)
        media = np.nanmean(suav, axis=0)
        ax.plot(rejilla, media, color=color, lw=2,
                label='{} (media de {} semilla{})'.format(nombre, len(runs), 's' if len(runs) > 1 else ''))
        if len(runs) > 1:
            ax.fill_between(rejilla, np.nanmin(suav, axis=0), np.nanmax(suav, axis=0),
                            color=color, alpha=0.25, lw=0, label='{}: mínimo-máximo'.format(nombre))
    if referencia is not None:
        ax.axhline(referencia, color='k', ls='--', lw=1.2,
                   label='política aleatoria ({:.1f})'.format(referencia))
    ax.set_ylabel(etiqueta_y)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')


def panel_causas(ax, nombre, runs, ventana):
    """Un panel por grupo: con varios grupos superpuestos no se lee."""
    rejilla = rejilla_grupo(runs)
    causas = sorted({c for run in runs for c in run['causas']})
    for causa in causas:
        series = []
        for run in runs:
            ind = np.array([1.0 if c == causa else 0.0 for c in run['causas']])
            series.append(en_rejilla(run['pasos'], media_movil(ind, ventana), rejilla))
        ax.plot(rejilla, 100.0 * np.nanmean(series, axis=0),
                color=COLORES_CAUSA.get(causa, 'k'), lw=1.6, label=causa)
    ax.set_title(nombre, fontsize=9, loc='left')
    ax.set_ylabel('Episodios que terminan\npor cada causa (%)')
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)


def main():
    ap = argparse.ArgumentParser(description='Curvas de aprendizaje de la práctica 1.')
    ap.add_argument('--grupo', action='append', required=True,
                    help='nombre=patrón de directorios (o monitor.csv), uno por semilla')
    ap.add_argument('--aleatoria', default=None, help='directorio o monitor.csv de la política aleatoria')
    ap.add_argument('--ventana', type=int, default=10, help='media móvil, en episodios')
    ap.add_argument('--paso-s', type=float, default=None,
                    help='duración media de un paso en s, para indicarla en el eje de la duración')
    ap.add_argument('--titulo', default='')
    ap.add_argument('--salida', default='curvas')
    args = ap.parse_args()

    grupos = []
    for g in args.grupo:
        nombre, _, patron = g.partition('=')
        runs = [leer_monitor(r) for r in expandir(patron)]
        runs = [r for r in runs if len(r['r'])]
        if not runs:
            raise SystemExit('Sin datos para el grupo ' + g)
        grupos.append((nombre, runs))

    ref = None
    if args.aleatoria:
        ref = float(np.mean(leer_monitor(args.aleatoria)['r']))

    n = 2 + len(grupos)
    fig, ejes = plt.subplots(n, 1, figsize=(8, 2.6 * n + 0.6), sharex=True)
    panel(ejes[0], grupos, 'r', args.ventana, 'Recompensa por episodio\n(adimensional)', ref)
    et_l = 'Duración del episodio\n(pasos{})'.format(
        ', 1 paso ≈ {:.2f} s'.format(args.paso_s) if args.paso_s else '')
    panel(ejes[1], grupos, 'l', args.ventana, et_l)
    for ax, (nombre, runs) in zip(ejes[2:], grupos):
        panel_causas(ax, nombre, runs, args.ventana)
    ejes[-1].set_xlabel('Pasos del entorno (llamadas a step())')
    n_sem = ', '.join('{} {} semilla{}'.format(n, len(r), 's' if len(r) > 1 else '') for n, r in grupos)
    fig.suptitle('{}\n{}\nmedia móvil de {} episodios; banda = mínimo-máximo'.format(
        args.titulo, n_sem, args.ventana), fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.salida) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig('{}.{}'.format(args.salida, ext), dpi=150)
    print('Figura guardada en {}.png y {}.pdf'.format(args.salida, args.salida))
    for nombre, runs in grupos:
        for run in runs:
            print('  {}: {} episodios, {} pasos, recompensa media de los últimos {} = {:.2f}'.format(
                run['ruta'], len(run['r']), int(run['pasos'][-1]), args.ventana,
                float(np.mean(run['r'][-args.ventana:]))))
    if ref is not None:
        print('  referencia aleatoria: {:.2f}'.format(ref))


if __name__ == '__main__':
    main()
