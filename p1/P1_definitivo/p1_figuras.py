#!/usr/bin/env python3
"""
Práctica 1 — Figuras de la caracterización y de las trayectorias.

Lee los CSV que dejan las herramientas y dibuja, con código propio:

  caracterizacion   tamaño del blob frente a la separación real, con el
                    modelo d = k·size^(-p) de p1_config.json, y x del blob
                    frente a la desviación real, con el campo de visión
  tilt              altura de la pelota en la imagen frente a la distancia,
                    para cada inclinación de la cámara (p1_barrido_tilt.csv)
  trayectorias      línea central del anillo y trayectorias del seguidor y
                    de la pelota en los episodios de evaluación indicados

No usa ROS 2: se ejecuta en Windows (python p1_figuras.py ...).

    python p1_figuras.py caracterizacion
    python p1_figuras.py tilt
    python p1_figuras.py trayectorias p1_eval/sac_C3_s0_C3_*_ep1.csv
"""

import argparse
import csv
import glob
import math
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from p1_comun import cargar_config, ModeloCamara  # noqa: E402

SALIDA = 'figuras'


def leer(ruta):
    with open(ruta, newline='') as f:
        return list(csv.DictReader(f))


def num(fila, campo):
    v = fila.get(campo, '')
    return float(v) if v not in ('', None) else math.nan


def guardar(fig, nombre):
    os.makedirs(SALIDA, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(SALIDA, '{}.{}'.format(nombre, ext)), dpi=150, bbox_inches='tight')
    print('Figura en {}/{}.png y .pdf'.format(SALIDA, nombre))


def fig_caracterizacion(args):
    cfg = cargar_config(args.config)
    cam = ModeloCamara(cfg)
    filas = [f for f in leer(args.csv) if f['visible'] == '1']
    sep = np.array([num(f, 'separacion_m') for f in filas])
    size = np.array([num(f, 'size') for f in filas])
    x = np.array([num(f, 'x') for f in filas])
    desv = np.array([num(f, 'desviacion_deg') for f in filas])

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.scatter(sep, size, s=18, label='lecturas (imagen nueva)')
    d = np.linspace(0.15, 1.2, 200)
    a.plot(d, (d / cam.k) ** (-1.0 / cam.p), 'k-',
           label='d = {:.3f}·size$^{{-{:.3f}}}$'.format(cam.k, cam.p))
    banda = cfg['protocolo']['banda_m']
    a.axvspan(banda[0], banda[1], color='tab:green', alpha=0.12, label='banda objetivo')
    a.set_yscale('log')
    a.set_xlabel('Separación real seguidor-pelota (m)')
    a.set_ylabel('Área del blob, size (píxeles, escala log.)')
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=8)

    b.scatter(desv, x, s=18, label='lecturas')
    th = np.linspace(-cam.fov / 2, cam.fov / 2, 100)
    signo = 1
    if len(desv) and np.nanmean(np.sign(desv) * np.sign(x - 50)) < 0:
        signo = -1
    b.plot(np.degrees(th), 50 + signo * 50 * np.tan(th) / cam.tan_semi, 'k-',
           label='x = 50 ± 50·tan(θ)/tan({:.1f}°)'.format(math.degrees(cam.fov) / 2))
    b.axvspan(-cfg['protocolo']['desviacion_ok_deg'], cfg['protocolo']['desviacion_ok_deg'],
              color='tab:green', alpha=0.12, label='|θ| < 15°')
    b.set_xlabel('Desviación real θ (°)')
    b.set_ylabel('x del blob (0-100)')
    b.grid(alpha=0.3)
    b.legend(fontsize=8)
    fig.tight_layout()
    guardar(fig, 'caracterizacion')


def fig_tilt(args):
    filas = leer(args.csv)
    fig, a = plt.subplots(figsize=(6, 4))
    for t in sorted({num(f, 'tilt_deg') for f in filas}):
        sel = [f for f in filas if num(f, 'tilt_deg') == t and f['visible'] == '1']
        if not sel:
            a.plot([], [], 'o-', label='tilt {:.0f}°: no se ve'.format(t))
            continue
        d = np.array([num(f, 'separacion_m') for f in sel])
        y = np.array([num(f, 'y') for f in sel])
        o = np.argsort(d)
        a.plot(d[o], y[o], 'o-', label='tilt {:.0f}°'.format(t))
    a.invert_yaxis()
    a.set_ylim(102, -2)
    a.axhline(100, color='k', lw=0.8, ls=':')
    a.set_xlabel('Separación real (m)')
    a.set_ylabel('y de la pelota en la imagen (0 arriba, 100 abajo)')
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.tight_layout()
    guardar(fig, 'barrido_tilt')


def fig_trayectorias(args):
    cfg = cargar_config(args.config)
    linea = np.load(cfg['carril']['archivo_linea_central'])
    fig, a = plt.subplots(figsize=(7, 6))
    a.plot(np.r_[linea[:, 0], linea[0, 0]], np.r_[linea[:, 1], linea[0, 1]], color='0.6', lw=8,
           alpha=0.4, label='carril del guía (línea central)')
    for ruta in [r for patron in args.csv for r in sorted(glob.glob(patron))]:
        filas = [f for f in leer(ruta) if f['seg_x'] != '']
        sx = [num(f, 'seg_x') for f in filas]
        sz = [num(f, 'seg_z') for f in filas]
        px = [num(f, 'pelota_x') for f in filas]
        pz = [num(f, 'pelota_z') for f in filas]
        nombre = os.path.basename(ruta).replace('.csv', '')
        a.plot(px, pz, '--', lw=1, label='pelota · ' + nombre)
        a.plot(sx, sz, '-', lw=1.5, label='seguidor · ' + nombre)
        a.plot(sx[0], sz[0], 'ko', ms=4)
    a.set_aspect('equal')
    a.set_xlabel('x (m)')
    a.set_ylabel('z (m)')
    a.grid(alpha=0.3)
    a.legend(fontsize=7, loc='best')
    fig.tight_layout()
    guardar(fig, args.nombre or 'trayectorias')


def main():
    ap = argparse.ArgumentParser(description='Figuras de caracterización y trayectorias.')
    ap.add_argument('figura', choices=['caracterizacion', 'tilt', 'trayectorias'])
    ap.add_argument('csv', nargs='*')
    ap.add_argument('--config', default='p1_config.json')
    ap.add_argument('--nombre', default=None, help='nombre del archivo de la figura de trayectorias')
    args = ap.parse_args()
    if args.figura == 'caracterizacion':
        args.csv = args.csv[0] if args.csv else 'p1_caracterizacion.csv'
        fig_caracterizacion(args)
    elif args.figura == 'tilt':
        args.csv = args.csv[0] if args.csv else 'p1_barrido_tilt.csv'
        fig_tilt(args)
    else:
        fig_trayectorias(args)


if __name__ == '__main__':
    main()
