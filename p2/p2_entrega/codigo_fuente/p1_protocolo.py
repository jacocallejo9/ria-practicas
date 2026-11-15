#!/usr/bin/env python3
"""
Práctica 1 — Registro y métricas del protocolo de evaluación (anexo B.4).

p1_seguidor.py --evaluar escribe un CSV por episodio con una fila por paso
(RegistroEvaluacion). Este módulo lee esos CSV y calcula, por política,
por configuración del guía y en conjunto, las cuatro métricas del
protocolo:

  Completados   número y porcentaje de episodios completados (120 s sin fallo)
  Desenlaces    número de episodios terminados por cada causa de B.3
  Separación    % de pasos dentro de la banda objetivo y separación media
  Desviación    media de |desviación| en grados y % de pasos por debajo de 15°

y, si había línea central, el % de pasos en el carril (no es métrica del
protocolo: el carril se juzga en los vídeos, pero ayuda a analizarlos).

No usa ROS 2, así que las tablas se pueden rehacer fuera del contenedor:

    python3 p1_protocolo.py p1_eval/*.csv
    python3 p1_protocolo.py p1_eval/*.csv --markdown > tabla.md
"""

import argparse
import csv
import glob
import os
from collections import Counter, OrderedDict

import numpy as np

CAMPOS = ['configuracion', 'politica', 'episodio', 'paso', 't_s',
          'seg_x', 'seg_z', 'pelota_x', 'pelota_z', 'guia_x', 'guia_z',
          'separacion_m', 'desviacion_deg', 'error_lateral_m',
          'en_banda', 'desviacion_ok', 'en_carril', 'visible', 'x_blob',
          'v_ms', 'w_rads', 'causa']

# Orden en que se presentan los desenlaces
DESENLACES = ['completado', 'colision', 'guia_perdido', 'sin_pelota']


class RegistroEvaluacion:
    """CSV con una fila por paso. La columna 'causa' sólo se rellena en la
    última fila del episodio, con su desenlace."""

    def __init__(self, ruta):
        os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
        self.ruta = ruta
        self._f = open(ruta, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow(CAMPOS)

    def fila(self, **valores):
        fila = []
        for c in CAMPOS:
            v = valores.get(c, '')
            if isinstance(v, bool):
                v = int(v)
            elif isinstance(v, float):
                v = '' if np.isnan(v) else round(v, 5)
            elif v is None:
                v = ''
            fila.append(v)
        self._w.writerow(fila)

    def cerrar(self):
        self._f.close()


# =====================================================================
# Lectura
# =====================================================================

def leer_episodios(rutas):
    """Lista de episodios; cada uno es un dict con sus filas y metadatos."""
    episodios = []
    for ruta in rutas:
        with open(ruta, newline='') as f:
            filas = list(csv.DictReader(f))
        por_ep = OrderedDict()
        for fila in filas:
            por_ep.setdefault(fila['episodio'], []).append(fila)
        for ep, fs in por_ep.items():
            causa = fs[-1]['causa'] or 'interrumpido'
            episodios.append({'archivo': ruta, 'episodio': ep, 'filas': fs,
                              'politica': fs[0]['politica'],
                              'configuracion': fs[0]['configuracion'],
                              'desenlace': causa})
    return episodios


def _columna(filas, nombre):
    vals = [f[nombre] for f in filas if f[nombre] != '']
    return np.array(vals, dtype=float)


# =====================================================================
# Métricas
# =====================================================================

def metricas(episodios):
    """Métricas del anexo B.4 sobre un conjunto de episodios.

    Los porcentajes de pasos se calculan sobre todos los pasos de todos
    los episodios del conjunto juntos.
    """
    n = len(episodios)
    desenlaces = Counter(e['desenlace'] for e in episodios)
    filas = [f for e in episodios for f in e['filas']]
    sep = _columna(filas, 'separacion_m')
    banda = _columna(filas, 'en_banda')
    desv = np.abs(_columna(filas, 'desviacion_deg'))
    desv_ok = _columna(filas, 'desviacion_ok')
    carril = _columna(filas, 'en_carril')
    return {
        'episodios': n,
        'completados': desenlaces.get('completado', 0),
        'pct_completados': 100.0 * desenlaces.get('completado', 0) / n if n else float('nan'),
        'desenlaces': dict(desenlaces),
        'pasos': len(filas),
        'pct_en_banda': 100.0 * banda.mean() if banda.size else float('nan'),
        'separacion_media': sep.mean() if sep.size else float('nan'),
        'desviacion_media': desv.mean() if desv.size else float('nan'),
        'pct_desviacion_ok': 100.0 * desv_ok.mean() if desv_ok.size else float('nan'),
        'pct_en_carril': 100.0 * carril.mean() if carril.size else float('nan'),
        'duracion_media_s': np.mean([float(e['filas'][-1]['t_s']) for e in episodios]) if n else float('nan'),
    }


def agrupar(episodios):
    """{(política, configuración): [episodios]}, más (política, 'TODAS')."""
    grupos = OrderedDict()
    for e in sorted(episodios, key=lambda e: (e['politica'], e['configuracion'])):
        grupos.setdefault((e['politica'], e['configuracion']), []).append(e)
    for pol in sorted({e['politica'] for e in episodios}):
        todos = [e for e in episodios if e['politica'] == pol]
        if len({e['configuracion'] for e in todos}) > 1:
            grupos[(pol, 'TODAS')] = todos
    return grupos


def _texto_desenlaces(d):
    otros = [k for k in d if k not in DESENLACES]
    return ', '.join('{} {}'.format(k, d[k]) for k in DESENLACES + otros if d.get(k))


def imprimir_tabla(grupos, markdown=False):
    cab = ['Política', 'Conf.', 'Completados', 'Desenlaces', 'En banda',
           'Sep. media', '|Desv.| media', 'Desv. < 15°', 'En carril']
    filas = []
    for (pol, conf), eps in grupos.items():
        m = metricas(eps)
        filas.append([
            pol, conf,
            '{}/{} ({:.0f} %)'.format(m['completados'], m['episodios'], m['pct_completados']),
            _texto_desenlaces(m['desenlaces']),
            '{:.1f} %'.format(m['pct_en_banda']),
            '{:.3f} m'.format(m['separacion_media']),
            '{:.1f}°'.format(m['desviacion_media']),
            '{:.1f} %'.format(m['pct_desviacion_ok']),
            '' if np.isnan(m['pct_en_carril']) else '{:.1f} %'.format(m['pct_en_carril']),
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


def main():
    ap = argparse.ArgumentParser(description='Métricas del anexo B a partir de los CSV de evaluación.')
    ap.add_argument('csv', nargs='+', help='archivos o patrones (p1_eval/*.csv)')
    ap.add_argument('--markdown', action='store_true')
    args = ap.parse_args()
    rutas = sorted({r for patron in args.csv for r in (glob.glob(patron) or [patron])})
    episodios = leer_episodios(rutas)
    if not episodios:
        raise SystemExit('No hay episodios en ' + ', '.join(args.csv))
    imprimir_tabla(agrupar(episodios), markdown=args.markdown)


if __name__ == '__main__':
    main()
