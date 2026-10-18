# Práctica 1 · Seguimiento de un vehículo en la ciudad (RIA 2026/27)

Un Robobo aprende por refuerzo (SAC y PPO de Stable-Baselines3) a seguir al vehículo guía
de *Follow City* en RoboboSim usando sólo sus sensores (blobs de la cámara e infrarrojos) a
través de ROS 2. La memoria está en `memoria/memoria.pdf` y la declaración de uso de IA en `Declaracion_IA/RIA_P1_Declaracion_IA.pdf`.

## Contenido

| Ruta | Qué es |
|---|---|
| `p1_seguidor_env.py` | Entorno de Gymnasium (observación, acción, paso sincronizado, terminación, reinicio) |
| `p1_recompensa.py` | Función de recompensa |
| `p1_seguidor.py` | Programa principal: calibración, línea base, entrenamiento, evaluación, ejecución |
| `p1_simulador.py` | Conexión directa con RoboboSim (verdad del terreno, reinicio, colocación) |
| `p1_protocolo.py` | Evaluador del anexo B: tablas de métricas a partir de los CSV de evaluación |
| `p1_curvas.py`, `p1_figuras.py` | Curvas de aprendizaje y figuras de caracterización |
| `p1_comun.py`, `p1_diagnostico.py` | Utilidades comunes y diagnóstico del simulador |
| `p1_config.json` | Parámetros de calibración, escenario y protocolo (todo lo que depende de la calibración) |
| `p1_linea_central.*` | Trayectoria del guía (línea central del carril) |
| `p1_caracterizacion.csv`, `p1_barrido_tilt.csv` | Medidas del sensor (figuras A1 y A2) |
| `p1_logs/definitivos/<run>/` | Cada entrenamiento: `modelo.zip`, `config.json` (configuración con la que se entrenó), `monitor.csv` (una fila por episodio) y, en SAC, `modelo_buffer.pkl` (sólo para continuar) |
| `p1_eval/definitivos/*.csv` | Registro por paso de cada episodio de evaluación |
| `figuras/` | Figuras de la memoria |
| `videos/` | Un vídeo por configuración con la política elegida evaluándose (`C1_sac_C1_s0.mp4`, `C2_sac_C2_s1.mp4`, `C3_sac_C3_s0.mp4`) y en `videos/registros/` el registro por paso de esos episodios. Los `.mp4` no están en GitHub (pesan más de 100 MB): van en el zip de entrega |
| `comandos.txt` | Chuleta de comandos |

Nombres de los entrenamientos: `<algoritmo>_<configuración>_s<semilla>`; `_ext` = entrenamiento
extendido; `aleatoria_Cx` = línea base.

## Puesta en marcha

1. RoboboSim en *Follow City*, guía configurado y *Apply*:
   C1 = Mode 1, 10-10 · C2 = Mode 1, 10-40 · C3 = Mode 2, 10-40. Ventana en primer plano
   (sin foco, RoboboSim congela la física).
2. Contenedor: `docker compose start` y dos terminales.
3. Terminal del puente (sólo el módulo de blobs):
   ```bash
   source /root/robobo_ws/install/setup.bash
   ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal -p modules:="['blob']"
   ```
4. Terminal de ejecución: `cd` a esta carpeta y `source /root/robobo_ws/install/setup.bash`.

## Reproducir

```bash
# Calibración (una vez; escribe en p1_config.json)
python3 p1_seguidor.py --calibrar-movimiento
python3 p1_seguidor.py --barrido-tilt --guardar
python3 p1_seguidor.py --caracterizar --guardar
python3 p1_seguidor.py --grabar-linea
python3 p1_seguidor.py --comprobar

# Línea base de entrenamiento (referencia de las curvas)
python3 p1_seguidor.py --aleatorio --episodios 10 --etiqueta C1

# Entrenamiento: SAC hasta el criterio de parada (éxito o estancamiento)
python3 p1_seguidor.py --entrenar --algoritmo sac --pasos 20000 --semilla 0 --etiqueta C1
# Presupuesto fijo (sin parada temprana) con la configuración de otro entrenamiento
python3 p1_seguidor.py --entrenar --algoritmo ppo --pasos 9431 --semilla 0 --etiqueta C1 --sin-parada --config p1_logs/definitivos/sac_C1_s0/config.json
# Continuar un entrenamiento
python3 p1_seguidor.py --entrenar --algoritmo sac --pasos 5000 --semilla 2 --etiqueta C2 --continuar p1_logs/definitivos/sac_C2_s2/modelo.zip

# Evaluación con el protocolo (usa la config.json guardada junto al modelo)
python3 p1_seguidor.py --evaluar --modelo p1_logs/definitivos/sac_C1_s0/modelo --configuracion C1 --episodios 3
python3 p1_seguidor.py --evaluar --modelo aleatoria --configuracion C1 --episodios 3

# Tablas y curvas (no necesitan ROS ni simulador; también funcionan en Windows)
python3 p1_protocolo.py "p1_eval/definitivos/*.csv"
python3 p1_curvas.py --grupo "SAC=p1_logs/definitivos/sac_C1_s[0-9]" --grupo "PPO=p1_logs/definitivos/ppo_C1_s0" --aleatoria p1_logs/definitivos/aleatoria_C1 --ventana 10 --paso-s 0.62 --titulo "Guía C1 (crucero 10-10)" --salida figuras/curvas_C1

# Ejecutar una política sin fin (vídeos)
python3 p1_seguidor.py --ejecutar --modelo p1_logs/definitivos/sac_C1_s0/modelo
```

## Políticas elegidas

| Configuración | Política | Evaluación |
|---|---|---|
| C1 (crucero 10-10) | `sac_C1_s0` | 3/3 completados, 100 % en banda, desviación media 2,1° |
| C2 (crucero 10-40) | `sac_C2_s1` | 3/3 completados, 98,2 % en banda, desviación media 4,6° |
| C3 (variable 10-40) | `sac_C3_s0` | Único SAC de C3 que cumplió el criterio de éxito (ver memoria) |

## Notas

- C2 se entrenó con `v_min = -0,05` m/s (está en su `config.json`); C1 y C3, con `v_min = 0`.
- `sac_C3_s1`: fallo de detección de la pelota desde el paso ≈3 615; se excluye de la media de
  C3. Las evaluaciones de C3 tienen 1-3 episodios por modelo.
- `ppo_C2_s0/monitor_hasta_fallo.csv`: registro completo del primer intento de PPO C2, que se
  cortó por un error y se continuó desde el checkpoint de 10 000 pasos.
- `cola_C1.log`: salida de los entrenamientos y evaluaciones de SAC C1 s1 y s2.
- Los `modelo_buffer.pkl` (~11 MB cada uno) sólo hacen falta para continuar un entrenamiento;
  se pueden quitar del .zip de entrega.
