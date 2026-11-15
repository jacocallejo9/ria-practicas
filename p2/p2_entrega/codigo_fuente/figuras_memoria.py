#!/usr/bin/env python3
"""
Figuras de la memoria de la P2 (no usa ROS ni el simulador).

Lee los registros por generación (evolucion.csv, especies.csv), los
genomas finales (.pkl) y los CSV del protocolo, y escribe en figuras/:

  evolucion.pdf     calidad del mejor y media de la población por
                    generación, una gráfica por configuración (media de
                    semillas; banda min-max del mejor y ±1 desv. típica
                    media de la población) con la referencia del
                    controlador a mano
  especies.pdf      tamaño de cada especie (áreas apiladas) en una
                    evolución por configuración
  completados.pdf   % de episodios completados y conexiones del mejor
  traza_C3.pdf      separación y desviación en el protocolo de C3
  red_<id>.dot      topología de cada controlador final (dot -Tpdf)

Uso, desde p2/p2_fran (la carpeta con p2_runs y p2_eval):
    python3 ../memoria_latex/figuras_memoria.py --salida ../memoria_latex/figuras
    for f in ../memoria_latex/figuras/red_*.dot; do dot -Tpdf "$f" -o "${f%.dot}.pdf"; done

El genoma se lee con neat-python; las gráficas, con matplotlib.
"""

import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

AZUL, NARANJA, GRIS, VERDE, ROJO = '#1f5fa8', '#d9771e', '#555555', '#2e8b57', '#b22222'

# Evoluciones de la solución de cada configuración (las semillas que se
# promedian en cada gráfica) y la referencia del controlador a mano.
GRUPOS = [
    ('C1: crucero 10-10', ['C1_v5_s1', 'C1_v5_s2', 'C1_v5_s3'],
     'referencia_C1_A.json', None),
    ('C2: crucero 10-40', ['C2_v5_p235_s1', 'C2_v5_p235_s3'],
     None, None),
    ('C3: variable 10-40', ['C3_v5_p235_s1', 'C3_v5_p235_s2'],
     'referencia_C3_A.json', None),
    ('C4: variable 10-40 con paradas', ['C4_v6_s1', 'C4_v6_s2', 'C4_v6_s3'],
     None, None),
]
EXTRA_C1 = 'C1_v6_s3'           # se dibuja aparte en C1 (otra configuración de NEAT)
ESPECIES = ['C1_v5_s2', 'C2_v5_p235_s3', 'C3_v5_p235_s2', 'C4_v6_s1']

# Nombres de las entradas en las figuras, en el orden de la red (claves -1 a -6).
ENTRADAS = ['error de distancia', 'posición de la pelota', 'cambio de distancia',
            'velocidad del guía', 'tiempo sin pelota', 'infrarrojo frontal']
SALIDAS = {0: 'velocidad', 1: 'giro'}


def coma(x):
    """Número con dos decimales y coma decimal."""
    return '{:.2f}'.format(x).replace('.', ',')


def fuente():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fuentes')
    for f in glob.glob(os.path.join(d, '*.ttf')) + glob.glob('/tmp/memo/fonts/*.ttf'):
        font_manager.fontManager.addfont(f)
    nombres = {f.name for f in font_manager.fontManager.ttflist}
    familia = 'Calibri' if 'Calibri' in nombres else ('Carlito' if 'Carlito' in nombres else 'DejaVu Sans')
    plt.rcParams.update({'font.family': familia, 'font.size': 9, 'axes.titlesize': 9,
                         'axes.labelsize': 9, 'legend.fontsize': 7.5, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'pdf.fonttype': 42})


def leer(run):
    with open(os.path.join('p2_runs', run, 'evolucion.csv')) as f:
        filas = list(csv.DictReader(f))
    r = {k: np.array([float(x[k]) for x in filas]) for k in filas[0]}
    with open(os.path.join('p2_runs', run, 'especies.csv')) as f:
        esp = list(csv.DictReader(f))
    ids = sorted({int(x['especie']) for x in esp})
    m = np.zeros((len(ids), len(filas)))
    for x in esp:
        i = int(x['indice'])
        if i < len(filas):
            m[ids.index(int(x['especie'])), i] = float(x['tam'])
    r['tam'], r['ids'] = m, ids
    return r


def ref(nombre):
    if not nombre:
        return None
    with open(os.path.join('p2_runs', nombre)) as f:
        return json.load(f)['calidad_media']


def pie(runs):
    pasos = sum(r['pasos'][-1] for r in runs)
    horas = sum(r['segundos'][-1] for r in runs) / 3600
    evals = sum(r['evaluaciones'][-1] for r in runs)
    return '{} semillas · {:.0f} evaluaciones · {:,.0f} pasos · {:.1f} h'.format(
        len(runs), evals, pasos, horas).replace(',', '.')


def fig_evolucion(salida):
    fig, ejes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharey=True)
    for ax, (titulo, nombres, nref, _) in zip(ejes.flat, GRUPOS):
        runs = [leer(n) for n in nombres]
        n = min(len(r['cal_mejor']) for r in runs)
        g = np.arange(n)
        mej = np.array([r['cal_mejor'][:n] for r in runs])
        med = np.array([r['cal_media'][:n] for r in runs])
        des = np.array([r['cal_desv'][:n] for r in runs])
        ax.fill_between(g, mej.min(0), mej.max(0), color=AZUL, alpha=0.18, lw=0)
        ax.plot(g, mej.mean(0), color=AZUL, lw=1.6, label='mejor individuo (media, mínimo y máximo)')
        ax.fill_between(g, np.clip(med.mean(0) - des.mean(0), 0, None), med.mean(0) + des.mean(0),
                        color=NARANJA, alpha=0.15, lw=0)
        ax.plot(g, med.mean(0), color=NARANJA, lw=1.4, label='media de la población (y desviación típica)')
        if nombres[0].startswith('C1'):
            e = leer(EXTRA_C1)
            ax.plot(np.arange(len(e['cal_media'])), e['cal_media'], color=VERDE, lw=1.1, ls='--',
                    label='media de la población, v6 s3')
        v = ref(nref)
        if v is not None:
            ax.axhline(v, color=GRIS, lw=1.0, ls=":", label="controlador a mano")
            ax.text(n - 1.2, v - 2, "a mano: {:.1f}".format(v).replace(".", ","), ha="right", va="top",
                    fontsize=7, color=GRIS)
        ax.set_title(titulo)
        print(nombres[0][:2], pie(runs))
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, n, 2 if n <= 12 else 4))
        ax.grid(alpha=0.25, lw=0.5)
    for ax in ejes[1]:
        ax.set_xlabel('generación')
    for ax in ejes[:, 0]:
        ax.set_ylabel('calidad (0 a 100)')
    h, l = ejes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.4, rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(salida, "evolucion.pdf"))
    plt.close(fig)


def fig_especies(salida):
    fig, ejes = plt.subplots(1, 4, figsize=(7.0, 1.75), sharey=True)
    cmap = plt.get_cmap('tab10')
    for ax, nombre in zip(ejes, ESPECIES):
        r = leer(nombre)
        n = r['tam'].shape[1]
        ax.stackplot(np.arange(n), r['tam'], colors=[cmap(i % 10) for i in range(len(r['ids']))],
                     alpha=0.85, lw=0)
        ax.set_title('{} ({} {})'.format(*nombre.replace('_p235', '').split('_')), fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, 20)
        ax.set_xlabel('generación')
        ax.grid(alpha=0.25, lw=0.5)
    ejes[0].set_ylabel('individuos')
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(salida, 'especies.pdf'))
    plt.close(fig)


def fig_completados(salida):
    fig, ejes = plt.subplots(1, 2, figsize=(7.0, 1.9))
    cols = [AZUL, NARANJA, VERDE, ROJO]
    for c, (titulo, nombres, _, _) in zip(cols, GRUPOS):
        runs = [leer(n) for n in nombres]
        n = min(len(r['cal_mejor']) for r in runs)
        pc = np.mean([r['pct_completados'][:n] for r in runs], axis=0)
        co = np.mean([r['conexiones_mejor'][:n] for r in runs], axis=0)
        et = titulo.split(':')[0]
        ejes[0].plot(np.arange(n), pc, color=c, lw=1.3, label=et)
        ejes[1].plot(np.arange(n), co, color=c, lw=1.3, label=et)
    ejes[0].set_ylabel('episodios completados (%)')
    ejes[1].set_ylabel('conexiones del mejor')
    for ax in ejes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('generación')
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, ncol=4, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(salida, 'completados.pdf'))
    plt.close(fig)


def leer_traza(patron):
    f = sorted(glob.glob(patron))[0]
    with open(f) as h:
        filas = list(csv.DictReader(h))
    t = np.array([float(x['t_s']) for x in filas])
    s = np.array([float(x['separacion_m']) for x in filas])
    d = np.array([float(x['desviacion_deg']) for x in filas])
    v = np.array([float(x['v_ms']) for x in filas])
    return t, s, d, v


def fig_traza(salida):
    fig, ejes = plt.subplots(1, 2, figsize=(7.0, 1.9))
    trazas = [('NEAT 411', 'p2_eval/neat_C3_v5_p235_s1_C3_*_ep1.csv', AZUL),
              ('a mano', 'p2_eval/manual_C3_*_ep1.csv', GRIS),
              ('aleatoria', 'p2_eval/aleatoria_C3_*_ep1.csv', ROJO)]
    for et, pat, c in trazas:
        t, s, d, _ = leer_traza(pat)
        ejes[0].plot(t, s, color=c, lw=1.0, label=et)
        ejes[1].plot(t, np.abs(d), color=c, lw=0.8, label=et)
    ejes[0].axhspan(0.20, 0.45, color=VERDE, alpha=0.10, lw=0)
    ejes[0].set_ylabel('separación (m)')
    ejes[0].set_ylim(0, 0.9)
    ejes[1].axhline(15, color=VERDE, lw=0.8, ls=':')
    ejes[1].set_ylabel('desviación (grados)')
    ejes[1].set_ylim(0, 40)
    for ax in ejes:
        ax.set_xlabel('t (s)')
        ax.set_xlim(0, 121)
        ax.grid(alpha=0.25, lw=0.5)
        leg = ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper right")
        for h in leg.get_lines():
            h.set_linewidth(2.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(salida, 'traza_C3.pdf'))
    plt.close(fig)


def dot_red(ruta_pkl, nombre, salida):
    # En el contenedor, matplotlib (apt) sólo funciona con el numpy 1.x del
    # sistema (PYTHONNOUSERSITE=1), y neat-python está en el site del
    # usuario: se añade al final del path sólo para leer el genoma.
    import site
    import sys
    if site.getusersitepackages() not in sys.path:
        sys.path.append(site.getusersitepackages())
    import neat  # noqa: F401
    with open(ruta_pkl, 'rb') as f:
        g = pickle.load(f)
    lin = ['digraph red {', '  rankdir=LR; nodesep=0.12; ranksep=0.9;',
           '  node [fontname="Calibri", fontsize=16, style=filled, penwidth=0.8];',
           '  edge [fontname="Calibri", fontsize=15, arrowsize=0.8];']
    usadas = {a for (a, b), c in g.connections.items() if c.enabled}
    lin.append('  { rank=same;')
    for i, e in enumerate(ENTRADAS):
        k = -(i + 1)
        col = '#dbe8f6' if k in usadas else '#f2f2f2'
        fc = 'black' if k in usadas else '#aaaaaa'
        lin.append('    i{} [label="{}", shape=box, fillcolor="{}", fontcolor="{}"];'.format(
            -k, e, col, fc))
    lin.append('  }')
    for k, n in g.nodes.items():
        if k in SALIDAS:
            lin.append('  n{} [label="{}\\nsesgo {}", shape=doublecircle, fillcolor="#fbe3cc"];'.format(
                k, SALIDAS[k], coma(n.bias)))
        else:
            lin.append('  n{} [label="oculta\\nsesgo {}", shape=circle, fillcolor="#e6e6e6"];'.format(
                k, coma(n.bias)))
    for (a, b), c in sorted(g.connections.items()):
        if not c.enabled:
            continue
        src = 'i{}'.format(-a) if a < 0 else 'n{}'.format(a)
        col = AZUL if c.weight > 0 else ROJO
        w = 0.6 + 1.4 * min(abs(c.weight), 2.5) / 2.5
        lin.append('  {} -> n{} [label="{}", color="{}", fontcolor="{}", penwidth={:.2f}];'.format(
            src, b, coma(c.weight), col, col, w))
    lin.append('}')
    with open(os.path.join(salida, 'red_{}.dot'.format(nombre)), 'w') as f:
        f.write('\n'.join(lin) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--salida', default='figuras')
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    fuente()
    fig_evolucion(a.salida)
    fig_especies(a.salida)
    fig_completados(a.salida)
    fig_traza(a.salida)
    for ruta, nombre in [('p2_runs/C1_v5_s2/genoma_final.pkl', '160'),
                         ('p2_runs/C2_v5_p235_s3/genoma_final.pkl', '411'),
                         ('p2_runs/C4_v6_s1/genoma_final.pkl', '446')]:
        dot_red(ruta, nombre, a.salida)
    print('Figuras en', a.salida)


if __name__ == '__main__':
    main()
