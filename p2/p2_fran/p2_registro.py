#!/usr/bin/env python3
"""
Práctica 2 — Registro por generación (reporter de NEAT-Python).

Escribe, junto a los resultados de cada evolución:

  evolucion.csv   una fila por generación: calidad del mejor, media,
                  desviación y mínimo; especies; nodos y conexiones del
                  mejor; evaluaciones, pasos y segundos acumulados (se
                  continúan al reanudar); % de episodios completados y
                  segundos de la generación.
  especies.csv    tamaño de cada especie en cada generación (formato largo)
  genomas/mejor_genNNN.pkl   mejor genoma de cada generación

No guarda archivos abiertos ni el entorno (el Checkpointer lo serializa con
pickle). Los contadores globales los actualiza p2_neat.py.
"""

import csv
import os
import pickle
import statistics
import time

import neat

CONTADOR = {"pasos": 0, "evaluaciones": 0, "episodios": 0, "completados": 0}
ESTADO = {"generacion": 0}


class RegistroGeneracion(neat.reporting.BaseReporter):
    CAMPOS = ["generacion", "cal_mejor", "cal_media", "cal_desv", "cal_min", "especies",
              "nodos_mejor", "conexiones_mejor", "evaluaciones", "pasos", "segundos",
              "pct_completados", "seg_generacion"]

    def __init__(self, ruta_csv, carpeta_genomas):
        self.ruta, self.carpeta = ruta_csv, carpeta_genomas
        self.ruta_esp = os.path.join(os.path.dirname(ruta_csv) or '.', 'especies.csv')
        os.makedirs(carpeta_genomas, exist_ok=True)
        self.base = {"evaluaciones": 0, "pasos": 0, "segundos": 0.0}
        self.fila = 0
        if os.path.exists(ruta_csv):                     # se reanuda una evolución
            with open(ruta_csv) as f:
                filas = list(csv.DictReader(f))
            if filas:
                self.base = {k: float(filas[-1][k]) for k in self.base}
                self.fila = len(filas)
        else:
            with open(ruta_csv, "w", newline="") as f:
                csv.writer(f).writerow(self.CAMPOS)
        if not os.path.exists(self.ruta_esp):
            with open(self.ruta_esp, "w", newline="") as f:
                csv.writer(f).writerow(["indice", "generacion", "especie", "tam"])
        self.inicio = dict(CONTADOR)
        self.t0 = time.time()
        self.gen = 0
        self.t_gen = time.time()
        self.c_gen = dict(CONTADOR)

    def start_generation(self, generation):
        self.gen = generation
        ESTADO["generacion"] = generation
        self.t_gen = time.time()
        self.c_gen = dict(CONTADOR)

    def post_evaluate(self, config, population, species, best_genome):
        cal = [g.fitness for g in population.values()]
        nodos, conexiones = best_genome.size()
        eps = CONTADOR["episodios"] - self.c_gen["episodios"]
        comp = CONTADOR["completados"] - self.c_gen["completados"]
        fila = [self.gen, round(best_genome.fitness, 4), round(statistics.mean(cal), 4),
                round(statistics.pstdev(cal), 4), round(min(cal), 4),
                len(species.species), nodos, conexiones,
                int(self.base["evaluaciones"] + CONTADOR["evaluaciones"] - self.inicio["evaluaciones"]),
                int(self.base["pasos"] + CONTADOR["pasos"] - self.inicio["pasos"]),
                round(self.base["segundos"] + time.time() - self.t0, 1),
                round(100.0 * comp / eps, 1) if eps else 0.0,
                round(time.time() - self.t_gen, 1)]
        with open(self.ruta, "a", newline="") as f:
            csv.writer(f).writerow(fila)
        with open(self.ruta_esp, "a", newline="") as f:
            w = csv.writer(f)
            for sid, s in species.species.items():
                w.writerow([self.fila, self.gen, sid, len(s.members)])
        self.fila += 1
        with open(os.path.join(self.carpeta, "mejor_gen{:03d}.pkl".format(self.gen)), "wb") as f:
            pickle.dump(best_genome, f)
