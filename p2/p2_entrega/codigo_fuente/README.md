# Práctica 2: seguimiento de un vehículo con neuroevolución (NEAT)

Jacobo Callejo Iribarren · Francisco Novo Rego

Este código evoluciona con NEAT-Python 2.0 un controlador para que el Robobo seguidor circule detrás del guía en Follow City (RoboboSim). Lo evalúa con el protocolo del anexo B y lo compara con la política SAC de la P1.

## Archivos

| Archivo | Contenido |
|---|---|
| `p2_neat.py` | Programa principal: controlador a mano, medición de tiempos, evolución, reanudación, reevaluación y ejecución del controlador final. |
| `p2_calidad.py` | Función de calidad, `F = 20·n/N + 70·Σq/N + 10·[completado]`. `python3 p2_calidad.py` imprime su tabla de valores. |
| `p2_registro.py` | Registro por generación (`evolucion.csv`, `especies.csv`, mejor genoma de cada generación). |
| `p2_protocolo.py` | Protocolo del anexo B (C1-C4, con la métrica de paradas de C4) para redes NEAT, el controlador a mano, la política aleatoria y los modelos de la P1. También genera las tablas. |
| `p2_figuras.py` | Curvas de evolución, especiación y complejidad a partir de los CSV. No necesita ROS. |
| `p2_red.py` | Figura de la topología de una red evolucionada. No necesita ROS. |
| `figuras_memoria.py` | Figuras del anexo de la memoria (curvas, especies, episodios completados, redes y evaluación en C3). No necesita ROS. |
| `p1_seguidor_env.py`, `p1_comun.py`, `p1_simulador.py`, `p1_recompensa.py`, `p1_protocolo.py` | Entorno de Gymnasium y evaluador de la P1, reutilizados sin modificar. |
| `config-p2.ini` | Configuración de NEAT v5 (evoluciones C1_v5, C2_v5 y C3_v5). |
| `config-p2-v6.ini` | Configuración de NEAT v6, con menos mutación (C1_v6, C3_v6 y C4_v6). |
| `p1_config.json` | Calibración del simulador y parámetros del protocolo. |
| `p1_linea_central.npy` / `.csv` | Línea central del carril, para la calidad (término de carril). |

Cada evolución se guarda en `p2_runs/<nombre>/`. Esa carpeta contiene:
- `config-p2.ini`, con la semilla, y `meta.json`, con las entradas, los pasos, los episodios por individuo, la configuración del guía y el origen si es incremental;
- `evolucion.csv`, con una fila por generación;
- `especies.csv`, `evaluaciones.csv` (un episodio por fila), `arranques.csv` y `centinela.csv`;
- `chk/p2-k`, el punto de control de cada generación;
- `genomas/`, con el mejor genoma de cada generación;
- `genoma_final.pkl` y `genoma_final.json`, el controlador elegido por reevaluación.

Cada episodio del protocolo se guarda paso a paso en `p2_eval/`.

Los programas buscan `p2_runs/` y `p2_eval/` en la carpeta desde la que se ejecutan. Para usarlos con los resultados entregados, se copian juntos a la carpeta de trabajo del contenedor:

```bash
cp codigo_fuente/* /docker-ros/ws/
cp -r evidencias/p2_runs evidencias/p2_eval /docker-ros/ws/
```

## Requisitos y puesta en marcha

- La imagen del contenedor de la asignatura, con NEAT-Python 2.0 (`pip show neat-python`), ROS 2 con `robobo_ros2` y Stable-Baselines3. Stable-Baselines3 sólo se usa para ejecutar la política de la P1.
- RoboboSim 4.4.1 con el mundo Follow City y el multiplicador de velocidad a 1,0. Es el mismo que en la P1, en la evolución y en la evaluación.

**Antes de cada orden** hay que configurar el guía en RoboboSim, pulsar Apply y dejar la ventana del simulador en primer plano. Sin foco, el simulador congela la física.

| Configuración | Menú del guía en RoboboSim |
|---|---|
| C1 | Mode 1, velocidad 10-10 |
| C2 | Mode 1, velocidad 10-40 |
| C3 | Mode 2, velocidad 10-40 |
| C4 | Mode 3, velocidad 10-40 (variable con paradas) |

**Terminal 1**, el puente, sólo con el módulo de blobs:

```bash
source /opt/ros/robobo/setup.bash
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal -p modules:="['blob']"
```

**Terminal 2**, la ejecución, desde la carpeta de trabajo:

```bash
source /opt/ros/robobo/setup.bash
cd /docker-ros/ws
```

Las órdenes piden confirmar la configuración del guía con Enter. `--sin-pausa` lo omite, para encadenar órdenes con `;`.

## 1. Referencia y medición del coste

```bash
python3 p2_neat.py --manual --etiqueta C1 --pasos 150     # controlador a mano con la misma calidad -> p2_runs/referencia_C1_A.json
python3 p2_neat.py --medir --pasos 150                    # tiempo de reinicio y de paso, y estimación del coste de una generación
```

## 2. Reproducir las evoluciones

La semilla de NEAT se fija con `--semilla` y queda escrita en el `.ini` copiado a la carpeta de la evolución. Todas las evoluciones usan una población de 20 individuos, la formulación A de la calidad y reinicio de escena. Los episodios son de 150 pasos (unos 77 s), salvo en `C2_v5_p235` y `C3_v5_p235`, que son de 235 pasos (los 120 s del protocolo).

**C1 (Mode 1, 10-10).** Desde cero. El controlador final es el genoma 160, de la semilla 2:

```bash
python3 p2_neat.py --evolucionar --etiqueta C1 --semilla 1 --generaciones 20 --pasos 150 --salida p2_runs/C1_v5_s1
python3 p2_neat.py --evolucionar --etiqueta C1 --semilla 2 --generaciones 20 --pasos 150 --salida p2_runs/C1_v5_s2
python3 p2_neat.py --evolucionar --etiqueta C1 --semilla 3 --generaciones 20 --pasos 150 --salida p2_runs/C1_v5_s3
python3 p2_neat.py --evolucionar --etiqueta C1 --semilla 3 --generaciones 15 --pasos 150 --config-ini config-p2-v6.ini --salida p2_runs/C1_v6_s3
```

**C2 (Mode 1, 10-40).** Evolución incremental desde la población de la generación 12 de `C1_v5_s2`. El primer intento, con 150 pasos, no funcionó; con 235 pasos la semilla 3 dio el genoma 411, que es el controlador final:

```bash
python3 p2_neat.py --evolucionar --etiqueta C2 --semilla 2 --generaciones 30 --pasos 150 --desde p2_runs/C1_v5_s2/chk/p2-12 --salida p2_runs/C2_v5_s2
python3 p2_neat.py --evolucionar --etiqueta C2 --semilla 1 --generaciones 25 --pasos 235 --desde p2_runs/C1_v5_s2/chk/p2-12 --salida p2_runs/C2_v5_p235_s1
python3 p2_neat.py --evolucionar --etiqueta C2 --semilla 3 --generaciones 21 --pasos 235 --desde p2_runs/C1_v5_s2/chk/p2-12 --salida p2_runs/C2_v5_p235_s3
```

**C3 (Mode 2, 10-40).** Evolución incremental desde la población de la generación 18 de `C2_v5_p235_s3`. El controlador final sigue siendo el genoma 411:

```bash
python3 p2_neat.py --evolucionar --etiqueta C3 --semilla 1 --generaciones 10 --pasos 235 --desde p2_runs/C2_v5_p235_s3/chk/p2-18 --salida p2_runs/C3_v5_p235_s1
python3 p2_neat.py --evolucionar --etiqueta C3 --semilla 2 --generaciones 10 --pasos 235 --desde p2_runs/C2_v5_p235_s3/chk/p2-18 --salida p2_runs/C3_v5_p235_s2
```

También se intentó pasar directamente de C1 a C3, desde la población de la generación 12 de `C1_v6_s3`, sin éxito. Esa población es el origen de C4:

```bash
python3 p2_neat.py --evolucionar --etiqueta C3 --semilla 1 --generaciones 10 --pasos 150 --config-ini config-p2-v6.ini --desde p2_runs/C1_v6_s3/chk/p2-12 --salida p2_runs/C3_v6_s1
```

Las semillas 2 y 3 se lanzan igual, con `--semilla 2` y `--salida p2_runs/C3_v6_s2`, y con `--semilla 3` y `--salida p2_runs/C3_v6_s3`.

**C4 (Mode 3, 10-40).** Evolución incremental desde C3, con 2 episodios por individuo. El centinela está desactivado porque el controlador a mano no está diseñado para las paradas:

```bash
python3 p2_neat.py --evolucionar --etiqueta C4 --semilla 1 --generaciones 10 --pasos 150 --episodios-ind 2 --config-ini config-p2-v6.ini --desde p2_runs/C3_v6_s1/chk/p2-4 --salida p2_runs/C4_v6_s1 --centinela 0
python3 p2_neat.py --evolucionar --etiqueta C4 --semilla 2 --generaciones 10 --pasos 150 --episodios-ind 2 --config-ini config-p2-v6.ini --desde p2_runs/C3_v6_s2/chk/p2-4 --salida p2_runs/C4_v6_s2 --centinela 0
python3 p2_neat.py --evolucionar --etiqueta C4 --semilla 3 --generaciones 10 --pasos 150 --episodios-ind 2 --config-ini config-p2-v6.ini --desde p2_runs/C3_v6_s1/chk/p2-4 --salida p2_runs/C4_v6_s3 --centinela 0
```

**Protecciones durante la evolución:**
- Cada 5 generaciones, el controlador a mano (centinela) tiene que completar un episodio. Si no lo hace, la evolución se detiene antes de evaluar la generación, porque el simulador está degradado.
- En cada generación se comprueba el signo de la x del blob.
- Un arranque inválido se repite, y un congelamiento del simulador también; no se le asigna un 0 al individuo.

**Reanudar tras un corte** (caída del puente, centinela, Ctrl-C). Continúa desde el último punto de control, con los mismos pasos y episodios:

```bash
python3 p2_neat.py --reanudar --etiqueta C1 --semilla 1 --salida p2_runs/C1_v5_s1
```

**Elegir el controlador final.** Reevalúa con 3 episodios a los mejores candidatos de la evolución y guarda en `genoma_final.pkl` el que completa más episodios (y, a igualdad, el de mayor calidad media):

```bash
python3 p2_neat.py --reevaluar --etiqueta C1 --semilla 1 --salida p2_runs/C1_v5_s1 --candidatos 7
```

## 3. Ejecutar el controlador final de cada configuración

Los controladores se cargan de `genoma_final.pkl` de su carpeta, o de `mejor_genoma.pkl` si no hay final. Con `p2_protocolo.py` se ejecutan con el protocolo del anexo B: episodios de 120 s (dos en C4), y el resultado se guarda en `p2_eval/`.

| Configuración | Controlador final | Orden |
|---|---|---|
| C1 | genoma 160 (C1_v5_s2) | `python3 p2_protocolo.py --evaluar --run p2_runs/C1_v5_s2 --configuracion C1` |
| C2 | genoma 411 (C2_v5_p235_s3) | `python3 p2_protocolo.py --evaluar --run p2_runs/C2_v5_p235_s3 --configuracion C2` |
| C3 | genoma 411 (C3_v5_p235_s1) | `python3 p2_protocolo.py --evaluar --run p2_runs/C3_v5_p235_s1 --configuracion C3` |
| C4 | genoma 446 (C4_v6_s1) | `python3 p2_protocolo.py --evaluar --run p2_runs/C4_v6_s1 --configuracion C4` |

Los demás genomas finales se evalúan igual, cambiando la carpeta de `--run`.

**Para ejecutarlo de forma continua**, sin protocolo, por ejemplo para grabar un vídeo. Reinicia la escena cada vez que falla y se detiene con Ctrl-C:

```bash
python3 p2_neat.py --ejecutar --etiqueta C1 --semilla 1 --salida p2_runs/C1_v5_s1
```

**Líneas base y política de la P1.** Para la P1, el modelo de Stable-Baselines3 tiene que estar junto a su `config.json`:

```bash
python3 p2_protocolo.py --evaluar --manual    --configuracion C3
python3 p2_protocolo.py --evaluar --aleatoria --configuracion C3
python3 p2_protocolo.py --evaluar --modelo-p1 p1_modelos/sac_C1_s0/modelo.zip --configuracion C3
```

## 4. Tablas y figuras

Ninguna de estas órdenes necesita ROS ni el simulador.

```bash
python3 p2_protocolo.py p2_eval/*.csv                     # métricas del anexo B por política y configuración
python3 p2_protocolo.py p2_eval/*C4*.csv --paradas        # detalle de cada parada del guía en C4

python3 p2_figuras.py --run p2_runs/C1_v5_s1 --run p2_runs/C1_v5_s2 --run p2_runs/C1_v6_s3 \
    --referencia p2_runs/referencia_C1_A.json --titulo "C1" --salida figuras/p2_C1_final
python3 p2_figuras.py --run p2_runs/C4_v6_s1 --run p2_runs/C4_v6_s2 --run p2_runs/C4_v6_s3 \
    --titulo "C4" --salida figuras/p2_C4_v6
python3 p2_red.py --run p2_runs/C1_v5_s1                  # topología del genoma 152 -> figuras/p2_red_C1_v5_s1.png
```

Las figuras del anexo de la memoria se generan todas juntas. En el contenedor, el matplotlib del sistema necesita el numpy del sistema (por eso `PYTHONNOUSERSITE=1`), y las topologías se dibujan con `dot`:

```bash
PYTHONNOUSERSITE=1 python3 figuras_memoria.py --salida figuras
for f in figuras/red_*.dot; do dot -Tpdf "$f" -o "${f%.dot}.pdf"; done
```
