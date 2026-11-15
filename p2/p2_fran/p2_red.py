#!/usr/bin/env python3
"""
Práctica 2 — Figura de la topología de una red evolucionada (no usa ROS).

Basada en dibujar_red() de las indicaciones técnicas: red por capas con
Matplotlib, sin Graphviz. Verde = peso positivo, rojo = negativo; el grosor
es proporcional al peso y cada conexión lleva su valor. Bajo cada nodo, su
sesgo. Las conexiones desactivadas no se dibujan.

    python3 p2_red.py --run p2_runs/C1_v5_s1                 # genoma_final.pkl
    python3 p2_red.py --run p2_runs/C1_v6_s3 --genoma mejor_genoma.pkl
    python3 p2_red.py --run p2_runs/C1_v5_s1 --salida figuras/p2_red_C1_s1
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import neat  # noqa: E402

ENTRADAS = {1: ['visible', 'x', 'distancia', 'vel_dist', 'vel_guia', 'ir_frontal'],
            2: ['err_dist', 'x', 'vel_dist', 'vel_guia', 'sin_pelota', 'ir_frontal']}
SALIDAS = ['v', 'w']


def cargar(run, nombre_genoma):
    ini = os.path.join(run, 'config-p2.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, ini)
    ruta = os.path.join(run, nombre_genoma)
    if not os.path.exists(ruta) and nombre_genoma == 'genoma_final.pkl':
        ruta = os.path.join(run, 'mejor_genoma.pkl')      # igual que p2_protocolo.py
    with open(ruta, 'rb') as f:
        genoma = pickle.load(f)
    if isinstance(genoma, tuple):
        genoma = genoma[0]
    version = 2
    meta = os.path.join(run, 'meta.json')
    if os.path.exists(meta):
        with open(meta) as f:
            version = int(json.load(f).get('entradas_version', 1))
    return genoma, config, ENTRADAS[version], ruta


def dibujar_red(genoma, config, nombres_entrada, nombres_salida, ruta, titulo):
    gc = config.genome_config
    entradas, salidas = list(gc.input_keys), list(gc.output_keys)
    activas = [c.key for c in genoma.connections.values() if c.enabled]
    # Profundidad de cada nodo: camino más largo desde las entradas
    prof = {n: 0 for n in entradas}
    for _ in range(len(genoma.nodes) + 1):
        for (a, b) in activas:
            nueva = prof[a] + 1 if a in prof and a != b else -1
            if prof.get(b, -1) < nueva <= len(genoma.nodes):
                prof[b] = nueva
    ocultas = [n for n in prof if n not in entradas and n not in salidas]
    capas = [entradas]
    for p in sorted({prof[n] for n in ocultas}):
        capas.append(sorted(n for n in ocultas if prof[n] == p))
    capas.append(salidas)
    pos = {n: (i * 2.0, -(j - (len(c) - 1) / 2))
           for i, c in enumerate(capas) for j, n in enumerate(c)}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (a, b) in activas:
        if a in pos and b in pos:
            w = genoma.connections[(a, b)].weight
            ax.annotate('', xy=pos[b], xytext=pos[a], arrowprops=dict(
                arrowstyle='->', shrinkA=11, shrinkB=11,
                lw=0.6 + 2.4 * min(abs(w), 3) / 3,
                color='tab:green' if w > 0 else 'tab:red', alpha=0.85))
            # etiqueta del peso, a un tercio del camino para no solaparse en la salida
            xa, ya = pos[a]
            xb, yb = pos[b]
            ax.text(xa + 0.35 * (xb - xa), ya + 0.35 * (yb - ya), '{:+.2f}'.format(w),
                    fontsize=6.5, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.8))
    nombres = dict(zip(entradas, nombres_entrada))
    nombres.update(zip(salidas, nombres_salida))
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=420, color='lightgrey', edgecolor='black', zorder=3)
        if n in entradas:
            ax.text(x - 0.25, y, nombres[n], ha='right', va='center', fontsize=9)
        else:
            etiqueta = nombres.get(n, 'oculta')
            ax.text(x, y + 0.28, etiqueta, ha='center', fontsize=9,
                    fontweight='bold' if n in salidas else 'normal')
            ax.text(x, y - 0.32, 'sesgo {:+.2f}'.format(genoma.nodes[n].bias),
                    ha='center', fontsize=7, color='0.3')
    ax.set_title(titulo, fontsize=10)
    ax.axis('off')
    ax.margins(x=0.25, y=0.15)
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig('{}.{}'.format(ruta, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Topología de una red evolucionada.')
    ap.add_argument('--run', required=True, help='carpeta de la evolución')
    ap.add_argument('--genoma', default='genoma_final.pkl', help='.pkl dentro de la carpeta')
    ap.add_argument('--salida', default=None, help='ruta sin extensión (por defecto figuras/p2_red_<run>)')
    args = ap.parse_args()
    genoma, config, nombres, ruta_g = cargar(args.run, args.genoma)
    run = os.path.basename(os.path.normpath(args.run))
    salida = args.salida or os.path.join('figuras', 'p2_red_' + run)
    activas = sum(c.enabled for c in genoma.connections.values())
    ocultas = len(genoma.nodes) - len(config.genome_config.output_keys)
    titulo = '{} · genoma {} · {} oculta(s), {} conexiones activas'.format(
        run, genoma.key, ocultas, activas)
    dibujar_red(genoma, config, nombres, SALIDAS, salida, titulo)
    print('Figura: {}.png y .pdf ({}; {} parámetros: {} pesos y {} sesgos)'.format(
        salida, ruta_g, activas + len(genoma.nodes), activas, len(genoma.nodes)))


if __name__ == '__main__':
    main()
