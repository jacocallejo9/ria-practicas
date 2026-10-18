#!/usr/bin/env python3
"""
Práctica 1 — Herramientas, entrenamiento y evaluación del seguidor en
"Follow City".

Misma estructura que el ejemplo de la pared (p1_entrenar.py), sobre el
entorno p1_seguidor_env.py. Instrucciones completas en README_P1.md.

--------------------------------------------------------------------
PUESTA EN MARCHA
--------------------------------------------------------------------

  1 · RoboboSim (Windows): mundo "Follow City", velocidad x1. En su menú,
      el modo y las velocidades del guía (para la política obligatoria:
      Variable, 10-40) y Apply. LA VENTANA DEL SIMULADOR TIENE QUE ESTAR EN
      PRIMER PLANO: si pierde el foco, la física se congela (el entorno lo
      detecta y espera).

  2 · Contenedor: el puente del seguidor, sólo con el módulo de blobs (la
      verdad del terreno y el reinicio van por p1_simulador.py):
        bash p1_puentes.sh start
      o, en una terminal aparte:
        source /root/robobo_ws/install/setup.bash
        ros2 run robobo_ros2 robobo_container --ros-args \\
             -p ip:=host.docker.internal -p modules:="['blob']"

  3 · Contenedor (cd /docker-ros/ws y el mismo source), en este orden:

        bash p1_calibrar.sh        movimiento, tilt, cámara y línea central
        python3 p1_seguidor.py --comprobar                   check_env
        python3 p1_seguidor.py --aleatorio --episodios 10 --etiqueta C3
        python3 p1_seguidor.py --entrenar --algoritmo sac --pasos 30000 --semilla 0 --etiqueta C3
        python3 p1_seguidor.py --evaluar --modelo p1_logs/sac_C3_s0/modelo --configuracion C3
        python3 p1_seguidor.py --evaluar --modelo aleatoria --configuracion C3
        python3 p1_protocolo.py p1_eval/*.csv
        python3 p1_curvas.py --grupo "SAC=p1_logs/sac_C3_s*" --aleatoria p1_logs/aleatoria_C3

      p1_calibrar.sh ejecuta, en orden:
        --calibrar-movimiento        velocidad real/pedida y convención de rumbo
        --barrido-tilt --guardar     inclinación de la cámara
        --caracterizar --guardar     tamaño del blob frente a distancia y campo de visión
        --grabar-linea               línea central del carril y velocidad del guía

Opciones: --semilla N  --pasos-max N  --detalle  --config archivo.json
          --sin-simulador (robot real: reinicio a mano, sin recompensa)
"""

import argparse
import copy
import csv
import math
import os
import shutil
import time

import numpy as np

from p1_comun import (cargar_config, guardar_config, angulo_hacia, ConvencionRumbo,
                      ModeloCamara, ajustar_distancia, ajustar_fov, fallo_protocolo)

RUTA_LOGS = os.path.join('p1_logs', 'definitivos')   # los anteriores, en p1_logs/anteriores
RUTA_EVAL = os.path.join('p1_eval', 'definitivos')

# Hiperparámetros. Cada paso cuesta ~0,3 s de tiempo real, así que se
# ajustan los que dependen del número de pasos (ver la guía de la P1).
# gamma no está aquí: se toma de p1_config.json, porque el moldeado de la
# recompensa tiene que usar el mismo.
HIPER_SAC = dict(learning_rate=3e-4, buffer_size=100_000, learning_starts=1000,
                 batch_size=256, tau=0.005, train_freq=1, gradient_steps=1,
                 ent_coef='auto')
HIPER_PPO = dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
                 gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)


def sufijo(args):
    return '_' + args.etiqueta if args.etiqueta else ''


def nombre_run(args):
    return '{}{}_s{}'.format(args.algoritmo, sufijo(args), args.semilla)


def crear_env(args, modo='entrenamiento', config=None, pasos_max=None):
    from p1_seguidor_env import RoboboSeguidorEnv
    return RoboboSeguidorEnv(config=config or args.config, modo=modo,
                             usar_simulador=not args.sin_simulador,
                             pasos_max=pasos_max or args.pasos_max, verbose=args.detalle)


def lista_numeros(texto):
    return [float(v) for v in texto.split(',') if v.strip()]


def reiniciar_sin_arranque(env):
    """Reinicia la escena y deja al seguidor parado en su sitio.

    Tras el reinicio el guía espera a que el seguidor se acerque a menos de
    ~0,35-0,40 m de su pelota, así que mientras no se le acerque se pueden
    hacer medidas con el guía quieto.
    """
    env._detener()
    env.cliente.reiniciar()
    env._esperar(env.cfg['espera_reset_s'])
    env.esperar_simulador_activo()
    env.colocar_tilt()
    env._medidas()


# =====================================================================
# --calibrar-movimiento : velocidad real frente a la pedida, y rumbo
# =====================================================================

def modo_calibrar_movimiento(args):
    """Mide cuánto se mueve de verdad el seguidor con una orden conocida.

    La cinemática del puente (radio de rueda y vía) no coincide con la del
    simulador: el robot recorre más de lo que pide cmd_vel. La maniobra es
    recta (3 s) - giro en el sitio (2 s) - recta (3 s), y de ella salen:

      factor_v   velocidad lineal real / pedida (las dos rectas)
      factor_w   velocidad angular real / pedida (el giro)
      rumbo      signo y desfase de la rotación y del simulador, comparando
                 la rotación con la dirección en la que avanza en las
                 rectas (dos rumbos distintos, así que el signo queda fijado)

    Las órdenes dan velocidades de rueda enteras, porque el puente las
    trunca: v = 0,064 m/s -> 20 en las dos ruedas; w = 0,669 rad/s -> ±15.
    """
    env = crear_env(args, modo='protocolo')
    V_CMD, W_CMD = 0.064, 0.669
    muestras = []

    def mover(v_cmd, w_cmd, segundos, tramo):
        env._publicar_velocidad(v_cmd, w_cmd)
        t0 = time.time()
        while time.time() - t0 < segundos:
            env._esperar(0.05)
            x, z, yaw, _ = env.cliente.seguidor()
            muestras.append((time.time() - t0, x, z, yaw, tramo))
        env._detener()
        env._esperar(0.6)

    try:
        reiniciar_sin_arranque(env)
        print('Maniobra de calibración: recta, giro a la izquierda y recta...')
        mover(V_CMD, 0.0, 3.0, 'recta1')
        mover(0.0, W_CMD, 2.0, 'giro')
        mover(V_CMD, 0.0, 3.0, 'recta2')
    finally:
        env.close()

    # Se descartan las muestras repetidas (el simulador envía la posición a
    # su ritmo) para que la velocidad salga de posiciones distintas.
    m = np.array([r[:4] for r in muestras], dtype=float)
    tramos = np.array([r[4] for r in muestras])
    factores_v, yaws, rumbos = [], [], []
    for tramo in ('recta1', 'recta2'):
        s = m[(tramos == tramo) & (m[:, 0] >= 0.5)]          # régimen estacionario
        if len(s) < 5:
            continue
        dist = np.hypot(s[-1, 1] - s[0, 1], s[-1, 2] - s[0, 2])
        factores_v.append(dist / (s[-1, 0] - s[0, 0]) / V_CMD)
        for i in range(len(s) - 1):
            j = i + 1
            while j < len(s) and np.hypot(s[j, 1] - s[i, 1], s[j, 2] - s[i, 2]) < 0.02:
                j += 1
            if j < len(s):
                rumbos.append(angulo_hacia(s[i, 1:3], s[j, 1:3]))
                yaws.append(0.5 * (s[i, 3] + s[j, 3]))
    g = m[(tramos == 'giro') & (m[:, 0] >= 0.4)]
    factor_w = signo_giro = None
    if len(g) >= 5:
        dyaw = np.degrees(np.unwrap(np.radians(g[:, 3])))
        velocidad = math.radians(dyaw[-1] - dyaw[0]) / (g[-1, 0] - g[0, 0])
        factor_w = abs(velocidad) / W_CMD
        signo_giro = 1 if velocidad > 0 else -1

    cfg = cargar_config(args.config)
    if factores_v:
        fv = float(np.mean(factores_v))
        print('factor_v = {:.3f}  (rectas: {})'.format(
            fv, ', '.join('{:.3f}'.format(f) for f in factores_v)))
        cfg['accion']['factor_v'] = round(fv, 3)
    if factor_w is not None:
        print('factor_w = {:.3f}  (con w > 0, la rotación y {})'.format(
            factor_w, 'crece' if signo_giro > 0 else 'decrece'))
        cfg['accion']['factor_w'] = round(factor_w, 3)
    if len(rumbos) > 6:
        signo, offset, R = ConvencionRumbo.estimar(yaws, rumbos)
        print('Convención de rumbo: rumbo = {:+d}·rotation.y {:+.1f}°   (consistencia R = {:.3f})'
              .format(signo, offset, R))
        if R > 0.95:
            cfg['rumbo']['signo'] = signo
            cfg['rumbo']['offset_deg'] = round(offset, 1)
        else:
            print('  R bajo: no se guarda la convención.')
    if args.no_guardar:
        print('No se guarda (--no-guardar).')
    else:
        guardar_config(cfg, args.config)
        print('Guardado en {}.'.format(args.config))


# =====================================================================
# --barrido-tilt : dónde aparece la pelota según la inclinación
# =====================================================================

def modo_barrido_tilt(args):
    """Posición y tamaño de la pelota en la imagen para cada inclinación.

    Para cada tilt: reinicio de la escena, se fija la inclinación y se
    coloca al seguidor a cada una de las distancias pedidas por detrás de
    la pelota, de lejos a cerca (mientras está lejos el guía espera quieto),
    leyendo tres mensajes de blobs en cada una. La mejor inclinación es la
    que ve la pelota en más distancias; a igualdad, la que la deja más
    centrada en la banda objetivo. Con --guardar se escribe en
    camara.tilt_deg.
    """
    tilts = lista_numeros(args.tilts)
    distancias = sorted(lista_numeros(args.distancias), reverse=True)
    env = crear_env(args, modo='protocolo')
    filas = []
    try:
        for tilt in tilts:
            reiniciar_sin_arranque(env)
            env.colocar_tilt(tilt)
            for d in distancias:
                env.colocar_seguidor_detras(d, en_linea=False)
                env._esperar(0.4)
                for _ in range(3):
                    env._esperar_blob_nuevo()
                    m = env._medidas()
                    vis = env._visible()
                    x, y, size, _ = env._blob if env._blob else (math.nan,) * 4
                    filas.append([tilt, d, round(m['separacion'], 4), int(vis),
                                  x if vis else math.nan, y if vis else math.nan,
                                  size if vis else math.nan])
                print('tilt {:5.1f}  d {:.2f} m  sep real {:.3f}  visible {}  y {:5.1f}  size {:6.1f}'
                      .format(tilt, d, filas[-1][2], filas[-1][3], filas[-1][5], filas[-1][6]))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    ruta = 'p1_barrido_tilt.csv'
    with open(ruta, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tilt_deg', 'distancia_objetivo_m', 'separacion_m', 'visible', 'x', 'y', 'size'])
        w.writerows(filas)
    print('\n{} lecturas en {}'.format(len(filas), ruta))
    if not filas:
        return

    datos = np.array(filas, dtype=float)
    print('\ny media de la pelota en la imagen (0 arriba, 100 abajo; "-" = no se ve)')
    print('distancia ' + ''.join('  tilt {:5.1f}'.format(t) for t in tilts))
    for d in sorted(distancias):
        celdas = []
        for t in tilts:
            sel = (datos[:, 0] == t) & (datos[:, 1] == d) & (datos[:, 3] > 0)
            celdas.append('{:11.1f}'.format(np.mean(datos[sel, 5])) if sel.any() else '          -')
        print('  {:.2f} m '.format(d) + ' '.join(celdas))

    banda = cargar_config(args.config)['protocolo']['banda_m']
    centro = 0.5 * (banda[0] + banda[1])
    d_centro = min(distancias, key=lambda d: abs(d - centro))
    mejor = None
    for t in tilts:
        sel = datos[:, 0] == t
        vistas = len({d for d, v in zip(datos[sel, 1], datos[sel, 3]) if v > 0})
        cerca = sel & (datos[:, 1] == d_centro) & (datos[:, 3] > 0)
        y_c = np.mean(datos[cerca, 5]) if cerca.any() else 1e9
        clave = (vistas, -abs(y_c - 50))
        if mejor is None or clave > mejor[0]:
            mejor = (clave, t)
    print('\nInclinación recomendada: {:.0f}° (ve la pelota en {} de {} distancias)'.format(
        mejor[1], mejor[0][0], len(distancias)))
    if args.guardar:
        cfg = cargar_config(args.config)
        cfg['camara']['tilt_deg'] = mejor[1]
        guardar_config(cfg, args.config)
        print('Guardada en {} (camara.tilt_deg).'.format(args.config))


# =====================================================================
# --grabar-linea : línea central del carril y velocidad del guía
# =====================================================================

def modo_grabar_linea(args):
    """Graba la trayectoria del guía durante una vuelta completa.

    De la misma vuelta salen:
      1. La línea central del carril (la trayectoria del guía), que sirve
         para medir el carril y la calzada y para colocar al seguidor.
      2. La velocidad real del guía en m/s, para justificar v_max.
      3. La distancia entre el centro del guía y su pelota.

    Tras el reinicio y el arranque, el seguidor se queda parado detrás del
    punto de salida, en el carril del guía, así que el guía se lo
    encontraría al cerrar la vuelta. Por eso la grabación termina cuando el
    guía vuelve cerca del punto de partida o se acerca a menos de 0,35 m del
    seguidor; el tramo que falta (recto) se cierra uniendo los extremos. Los
    ratos con el simulador congelado no cuentan para la velocidad.
    """
    cfg = cargar_config(args.config)
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp['carril']['archivo_linea_central'] = ''      # no cargar una línea anterior
    env = crear_env(args, modo='protocolo', config=cfg_tmp)
    env.reset()                                          # escena + arranque: el guía echa a andar
    env._detener()

    g = env.cliente.guia()
    puntos, muestras, offsets = [g], [(time.time(), g[0], g[1])], []
    inicio = g
    seg = np.array(env.cliente.seguidor()[:2])
    recorrido, t0 = 0.0, time.time()
    print('Grabando la vuelta del guía (Ctrl-C para cortar)...')
    try:
        while time.time() - t0 < 1800:
            env._esperar(0.05)
            g = env.cliente.guia()
            muestras.append((time.time(), g[0], g[1]))
            if len(muestras) % 20 == 0 and cfg.get('objeto_pelota'):
                offsets.append(np.linalg.norm(g - env.cliente.pelota()))
            paso = np.linalg.norm(g - puntos[-1])
            if paso >= 0.01:                                # un punto cada 1 cm
                puntos.append(g)
                recorrido += paso
                if len(puntos) % 50 == 0:
                    print('  {:5d} puntos, {:5.2f} m'.format(len(puntos), recorrido), end='\r')
            if recorrido > 2.0 and (np.linalg.norm(g - inicio) < 0.05 or
                                    np.linalg.norm(g - seg) < 0.35):
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    # 1. Línea central
    p = np.array(puntos)
    hueco = float(np.linalg.norm(p[-1] - p[0]))
    n = max(int(hueco / 0.01), 0)
    if n > 1:
        cierre = p[-1] + np.outer(np.linspace(0, 1, n + 1)[1:-1], p[0] - p[-1])
        p = np.vstack([p, cierre])
    ruta = cfg['carril']['archivo_linea_central']
    np.save(ruta, p)
    with open(ruta.replace('.npy', '.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['x_m', 'z_m']] + p.tolist())
    print('\nLínea central: {} puntos, {:.2f} m de vuelta, hueco cerrado {:.2f} m -> {}'
          .format(len(p), recorrido + hueco, hueco, ruta))

    # 2. Velocidad del guía (desplazamiento en ventanas de ~0,5 s)
    m = np.array(muestras)
    vel, j = [], 0
    for i in range(len(m)):
        while j < len(m) and m[j, 0] - m[i, 0] < 0.5:
            j += 1
        if j >= len(m):
            break
        vel.append(np.hypot(m[j, 1] - m[i, 1], m[j, 2] - m[i, 2]) / (m[j, 0] - m[i, 0]))
    vel = np.array(vel)
    vel = vel[vel > 0.002]                                  # fuera los ratos congelados
    if len(vel):
        print('Velocidad del guía (m/s): mínima {:.3f}  p5 {:.3f}  mediana {:.3f}  p95 {:.3f}  '
              'máxima {:.3f}'.format(vel.min(), *np.percentile(vel, [5, 50, 95]), vel.max()))
        if cfg['accion']['v_max'] < 1.2 * vel.max():
            print('  AVISO: v_max = {:.2f} m/s apenas supera al guía: el seguidor no podría '
                  'recuperar distancia.'.format(cfg['accion']['v_max']))
    if offsets:
        print('Distancia centro del guía - pelota: {:.3f} m'.format(float(np.mean(offsets))))


# =====================================================================
# --caracterizar : blob frente a distancia y ángulo (apartado 5.1)
# =====================================================================

def modo_caracterizar(args):
    """Registra la pelota vista por el seguidor frente a la verdad del terreno.

    Caracterización ESTÁTICA: tras reiniciar la escena (el guía espera
    quieto mientras el seguidor está lejos) se coloca al seguidor a cada
    distancia de --distancias-car, de lejos a cerca, mirando a la pelota y
    girado ±--giro grados (para que la pelota caiga a un lado de la imagen y
    medir el campo de visión). En cada posición se toma la primera imagen
    NUEVA del detector (el blob cambia a ~1,8 Hz en RoboboSim): si se usaran
    todas las lecturas, se emparejarían tamaños viejos con distancias nuevas.
    Por debajo de ~0,37 m el guía arranca, así que esas medidas se toman con
    él ya en marcha (sólo mirando de frente).

    Ajusta d = k·size^(-p) y el campo de visión; con --guardar los
    escribe en p1_config.json. En el robot real se repite esta medida
    colocando la pelota a mano a distancias conocidas.
    """
    env = crear_env(args, modo='protocolo')
    filas, t0 = [], time.time()
    distancias = sorted(lista_numeros(args.distancias_car), reverse=True)
    try:
        reiniciar_sin_arranque(env)
        for d in distancias:
            giros = [0.0, -args.giro, args.giro] if d >= 0.40 else [0.0]
            for giro in giros:
                env.colocar_seguidor_detras(d, en_linea=False, giro_deg=giro)
                env._esperar(0.3)
                env.esperar_blob_fresco(1.0)           # una imagen posterior a la colocación
                m = env._medidas()
                vis = env._visible()
                x, y, size, fts = env._blob if env._blob else (math.nan,) * 4
                ir = max(env._ir[i] for i in env.cfg['ir']['indices']) if env._ir else 0
                filas.append([round(time.time() - t0, 3), round(m['separacion'], 4),
                              round(math.degrees(m['desviacion']), 2), round(m['error_lateral'], 4),
                              int(vis), x, y, size, fts, ir, d, giro])
                print('  d={:.2f} giro={:+5.1f}°  sep={:.3f} m  desv={:+6.1f}°  visible={:d}  '
                      'x={:5.1f}  y={:5.1f}  size={:7.1f}  ir={}'.format(
                          d, giro, m['separacion'], math.degrees(m['desviacion']), vis, x, y,
                          size, ir))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    ruta = 'p1_caracterizacion{}.csv'.format(sufijo(args))
    with open(ruta, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'separacion_m', 'desviacion_deg', 'error_lateral_m', 'visible',
                    'x', 'y', 'size', 'frame_ts', 'ir_max', 'distancia_objetivo_m', 'giro_deg'])
        w.writerows(filas)
    print('\n{} lecturas en {}'.format(len(filas), ruta))
    analizar_caracterizacion(filas, args)


def analizar_caracterizacion(filas, args):
    cfg = cargar_config(args.config)
    p = cfg['protocolo']
    datos = np.array([[r[1], r[2], r[4], r[5], r[6], r[7]] for r in filas], dtype=float)
    if not len(datos):
        return
    sep, desv, vis, x, y, size = datos.T
    # Rango útil: de la colisión a algo más allá de "guía perdido". Más lejos
    # la pelota ocupa unos pocos píxeles y el tamaño es casi todo ruido.
    ok = (vis > 0) & (size > 0) & (sep > p['colision_m']) & (sep < p['perdido_m'] + 0.3)
    print('\nSeparaciones con pelota visible: {:.3f} - {:.3f} m'.format(
        sep[ok].min() if ok.any() else math.nan, sep[ok].max() if ok.any() else math.nan))
    if ok.sum() < 10:
        print('Muy pocas lecturas con la pelota visible para ajustar nada.')
        return

    # Modelo de distancia
    k, pe, rmse = ajustar_distancia(size[ok], sep[ok])
    print('Ajuste d = k·size^(-p):  k = {:.3f}  p = {:.3f}  (error cuadrático medio '
          '{:.3f} m; estenopeica pura: p = 0,5)'.format(k, pe, rmse))
    cfg_nuevo = copy.deepcopy(cfg)
    cfg_nuevo['camara']['distancia_k'] = round(k, 4)
    cfg_nuevo['camara']['distancia_p'] = round(pe, 4)
    cfg_nuevo['camara'].pop('distancia_b', None)

    # Campo de visión (sólo lecturas con la pelota fuera del centro)
    xn = (x - cfg['camara']['x_centro']) / cfg['camara']['x_semiancho']
    okf = ok & (np.abs(xn) > 0.05) & (np.abs(desv) < 60)
    if okf.sum() >= 10:
        fov, signo, err = ajustar_fov(xn[okf], np.radians(desv[okf]))
        print('Campo de visión horizontal: {:.1f}°  (x positiva = desviación {}; error {:.1f}°)'
              .format(fov, 'positiva' if signo > 0 else 'negativa', err))
        cfg_nuevo['camara']['fov_horizontal_deg'] = round(fov, 1)
    else:
        print('No hay lecturas con la pelota desplazada en la imagen: no se ajusta el campo '
              'de visión (usa --oscilar).')

    # Tabla por tramos de 5 cm con el modelo nuevo
    cam = ModeloCamara(cfg_nuevo)
    print('\n distancia   lecturas  visible   size medio ± desv.   d estimada   y media')
    for lo in np.arange(0.10, 1.30, 0.05):
        sel = (sep >= lo) & (sep < lo + 0.05)
        if not sel.any():
            continue
        sv = sel & ok
        if sv.any():
            dest = np.mean([cam.distancia(s) for s in size[sv]])
            print(' {:.2f}-{:.2f}   {:6d}   {:5.0f} %   {:8.1f} ± {:6.1f}     {:.3f} m     {:5.1f}'
                  .format(lo, lo + 0.05, sel.sum(), 100 * vis[sel].mean(), size[sv].mean(),
                          size[sv].std(), dest, y[sv].mean()))
        else:
            print(' {:.2f}-{:.2f}   {:6d}   {:5.0f} %'.format(lo, lo + 0.05, sel.sum(),
                                                            100 * vis[sel].mean()))
    if args.guardar:
        guardar_config(cfg_nuevo, args.config)
        print('\nModelo de la cámara guardado en {}.'.format(args.config))
    else:
        print('\nPara guardarlo en {}, repetir con --guardar.'.format(args.config))


# =====================================================================
# --comprobar · --aleatorio
# =====================================================================

def modo_comprobar(args):
    from stable_baselines3.common.env_checker import check_env
    env = crear_env(args)
    try:
        check_env(env, warn=True)
        print('\nEl entorno cumple la interfaz de Gymnasium.')
    finally:
        env.close()


def modo_aleatorio(args):
    """Política aleatoria en el entorno de entrenamiento.

    Su recompensa media por episodio es la línea de referencia de las
    curvas de aprendizaje; se registra con Monitor igual que un
    entrenamiento, en p1_logs/aleatoria<_etiqueta>/monitor.csv.
    """
    from stable_baselines3.common.monitor import Monitor
    directorio = os.path.join(RUTA_LOGS, 'aleatoria' + sufijo(args))
    os.makedirs(directorio, exist_ok=True)
    env = Monitor(crear_env(args), directorio, info_keywords=('causa',))
    env.action_space.seed(args.semilla)
    totales = []
    try:
        for ep in range(args.episodios or 10):
            obs, info = env.reset(seed=args.semilla + ep)
            fin, total, pasos, banda = False, 0.0, 0, 0
            while not fin:
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                fin = term or trunc
                total += r
                pasos += 1
                banda += info.get('en_banda', False)
            totales.append(total)
            print('episodio {:2d} ({:7s})  pasos {:3d}  recompensa {:7.2f}  en banda {:5.1f} %  {}'
                  .format(ep + 1, info.get('reinicio', ''), pasos, total, 100 * banda / pasos,
                          info['causa']))
    finally:
        env.close()
    print('\nRecompensa media {:.2f} (desviación {:.2f}) -> {}/monitor.csv'.format(
        np.mean(totales), np.std(totales), directorio))


# =====================================================================
# --entrenar
# =====================================================================

def clase_sac_con_avance(a):
    """SAC cuyo calentamiento empieza avanzando, en vez de con acciones al azar.

    SAC llena su búfer con learning_starts acciones uniformes antes de
    aprender; en esta tarea casi todas son giros sin sentido que pierden al
    guía en pocos pasos. Durante los primeros a['pasos'] pasos del
    entrenamiento (sólo en un entrenamiento nuevo: al continuar ya se han
    superado) la acción es avanzar:
      - velocidad: uniforme en a['v'] (acción normalizada en [-1, 1]), para
        que el búfer tenga avances lentos y rápidos;
      - giro: si a['girar_hacia_pelota'], proporcional a la x de la pelota
        (componente 1 de la observación) para no perderla en las curvas, más
        un ruido uniforme de ±a['ruido_w']; si no, sólo el ruido.
    El resto del calentamiento (hasta learning_starts) sigue siendo aleatorio,
    y el aprendizaje no cambia. No se aplica a PPO: es on-policy y aprende de
    la probabilidad de sus propias acciones.
    """
    from stable_baselines3 import SAC

    class SACConAvance(SAC):
        def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
            if self.num_timesteps >= a['pasos']:
                return super()._sample_action(learning_starts, action_noise, n_envs)
            rng = np.random.default_rng()
            obs = np.asarray(self._last_obs).reshape(n_envs, -1)
            acciones = []
            for o in obs:
                v = rng.uniform(*a['v'])
                w = rng.uniform(-a['ruido_w'], a['ruido_w'])
                if a['girar_hacia_pelota'] and o[0] > 0.5:          # pelota visible
                    # x > 0: pelota a la derecha -> girar a la derecha (w < 0)
                    w -= a['ganancia_giro'] * float(o[1])
                acciones.append([v, np.clip(w, -1.0, 1.0)])
            unscaled_action = np.array(acciones, dtype=np.float32)
            scaled_action = self.policy.scale_action(unscaled_action)
            return self.policy.unscale_action(scaled_action), scaled_action

    return SACConAvance


def leer_historial(ruta_monitor):
    """(recompensa, causa) de los episodios ya registrados en un monitor.csv."""
    if not os.path.exists(ruta_monitor):
        return []
    with open(ruta_monitor, newline='') as f:
        f.readline()                                   # cabecera JSON de Monitor
        return [(float(x['r']), x.get('causa', '')) for x in csv.DictReader(f)]


def crear_parada_temprana(p, pasos_inicio=0, historial=()):
    """Callback de Stable-Baselines3 que corta el entrenamiento cuando basta.

    Usa los propios episodios de entrenamiento (no hay evaluaciones aparte:
    cada paso cuesta tiempo real). Se miran los últimos p['ventana_episodios']
    episodios, sin contar los truncados por simulador congelado, y se para:

      - por ÉXITO, si al menos p['pct_completos'] % de ellos llegan al límite
        de pasos sin fallar (300 pasos ≈ 3 min, más que los 120 s del
        protocolo). Durante el entrenamiento la política explora; la
        determinista de la evaluación suele hacerlo igual o mejor.
      - por ESTANCAMIENTO, si la media de recompensa de la ventana no ha
        superado su mejor valor en más de p['mejora_minima'] (fracción) en
        los últimos p['paciencia_pasos'] pasos.

    Nunca antes de p['pasos_minimos'] pasos del entrenamiento (en total,
    contando las partes anteriores si se ha reanudado con --continuar). Al
    reanudar, la ventana se rellena con los últimos episodios del historial
    (monitor.csv), para no tener que esperar otros 20 episodios.
    """
    from collections import deque
    from stable_baselines3.common.callbacks import BaseCallback

    class ParadaTemprana(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ventana = deque(maxlen=p['ventana_episodios'])
            for r, causa in historial:
                if causa != 'congelado':
                    self.ventana.append((r, causa == 'tiempo'))
            self.mejor, self.paso_mejor = None, pasos_inicio
            self.motivo = ''

        def _on_step(self):
            for info in self.locals.get('infos', []):
                ep = info.get('episode')
                if ep is None or ep.get('causa') == 'congelado':
                    continue
                self.ventana.append((ep['r'], ep.get('causa') == 'tiempo'))
            if len(self.ventana) < self.ventana.maxlen or self.num_timesteps < p['pasos_minimos']:
                return True
            media = float(np.mean([r for r, _ in self.ventana]))
            completos = 100.0 * np.mean([c for _, c in self.ventana])
            if self.mejor is None or media > self.mejor + abs(self.mejor) * p['mejora_minima']:
                self.mejor, self.paso_mejor = media, self.num_timesteps
            if completos >= p['pct_completos']:
                self.motivo = ('éxito: {:.0f} % de los últimos {} episodios completos (recompensa '
                               'media {:.1f})'.format(completos, len(self.ventana), media))
            elif self.num_timesteps - self.paso_mejor >= p['paciencia_pasos']:
                self.motivo = ('estancamiento: la recompensa media ({:.1f}) no mejora desde el paso '
                               '{}'.format(media, self.paso_mejor))
            if self.motivo:
                print('\n*** Parada temprana en el paso {} por {} ***'.format(
                    self.num_timesteps, self.motivo))
                return False
            return True

    return ParadaTemprana()


def modo_entrenar(args):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    cfg = cargar_config(args.config)
    nombre = nombre_run(args)
    directorio = os.path.join(RUTA_LOGS, nombre)
    os.makedirs(directorio, exist_ok=True)

    # Copia de la configuración (y de la línea central) con la que se
    # entrena, junto al modelo: así se puede reproducir y evaluar después.
    guardar_config(cfg, os.path.join(directorio, 'config.json'))
    linea = cfg['carril']['archivo_linea_central']
    if linea and os.path.exists(linea):
        shutil.copy(linea, directorio)

    env = crear_env(args, config=cfg)
    try:
        # Monitor escribe <directorio>/monitor.csv: una fila por episodio con
        # recompensa, longitud, tiempo y causa de terminación (curvas).
        env = Monitor(env, directorio, info_keywords=('causa',),
                      override_existing=not args.continuar)
        try:
            import tensorboard  # noqa: F401
            tb = directorio
        except ImportError:
            tb = None
        Algoritmo = SAC if args.algoritmo == 'sac' else PPO
        avance = cfg.get('calentamiento_avance', {})
        if args.algoritmo == 'sac' and avance.get('activo'):
            Algoritmo = clase_sac_con_avance(avance)
            if not args.continuar:
                print('Calentamiento: los primeros {} pasos, avanzar (v normalizada en {}, giro '
                      '{}hacia la pelota).'.format(avance['pasos'], avance['v'],
                                                   '' if avance['girar_hacia_pelota'] else 'no '))
        hiper = dict(HIPER_SAC if args.algoritmo == 'sac' else HIPER_PPO,
                     gamma=cfg['recompensa']['gamma'])

        if args.continuar:
            modelo = Algoritmo.load(args.continuar, env=env, tensorboard_log=tb)
            buffer = args.continuar.replace('.zip', '') + '_buffer.pkl'
            if args.algoritmo == 'sac' and os.path.exists(buffer):
                modelo.load_replay_buffer(buffer)
                print('Búfer de repetición cargado ({} transiciones).'.format(
                    modelo.replay_buffer.size()))
        else:
            modelo = Algoritmo('MlpPolicy', env, verbose=1, seed=args.semilla,
                               tensorboard_log=tb, **hiper)

        pasos0 = modelo.num_timesteps
        callbacks = [CheckpointCallback(save_freq=args.cada,
                                        save_path=os.path.join(directorio, 'checkpoints'),
                                        name_prefix='modelo')]
        parada = cfg.get('parada_temprana', {})
        if parada.get('activa') and not args.sin_parada:
            historial = leer_historial(os.path.join(directorio, 'monitor.csv')) if args.continuar else []
            callbacks.append(crear_parada_temprana(parada, pasos0, historial))
        print('\nEntrenando {} durante {} pasos como máximo (semilla {}) -> {}'.format(
            args.algoritmo.upper(), args.pasos, args.semilla, directorio))
        if len(callbacks) > 1:
            print('Parada temprana: {} % de {} episodios completos, o sin mejorar en {} pasos '
                  '(desde {} pasos). Se desactiva con --sin-parada.'.format(
                      parada['pct_completos'], parada['ventana_episodios'],
                      parada['paciencia_pasos'], parada['pasos_minimos']))
        print('Se puede interrumpir con Ctrl-C: el modelo se guarda igual.\n')
        t0 = time.time()
        try:
            modelo.learn(total_timesteps=args.pasos, callback=CallbackList(callbacks),
                         tb_log_name='tb', reset_num_timesteps=not args.continuar)
        except KeyboardInterrupt:
            print('\nEntrenamiento interrumpido por el usuario.')
        except RuntimeError as e:
            # P. ej. se apaga la pantalla y RoboboSim deja de publicar blobs:
            # se guarda lo aprendido para poder seguir con --continuar.
            print('\nEntrenamiento interrumpido por un error: {}'.format(e))
        ruta_modelo = os.path.join(directorio, 'modelo')
        modelo.save(ruta_modelo)
        if args.algoritmo == 'sac':
            modelo.save_replay_buffer(ruta_modelo + '_buffer')
        dur = time.time() - t0
        hechos = max(modelo.num_timesteps - pasos0, 1)
        print('\nModelo guardado en {}.zip'.format(ruta_modelo))
        print('Tiempo empleado: {:.1f} min  ({:.3f} s por paso)'.format(dur / 60.0, dur / hechos))
    finally:
        env.close()


# =====================================================================
# --evaluar : protocolo del anexo B
# =====================================================================

def config_del_modelo(args):
    """Configuración con la que ejecutar un modelo ya entrenado.

    La acción y la observación dependen de la configuración (w_max, v_max,
    factores, modelo de la cámara...): un modelo hay que ejecutarlo con la
    misma con la que se entrenó, que --entrenar guarda en
    p1_logs/<run>/config.json. Se usa ésa salvo que se pase --config.
    """
    if args.config is not None:
        return args.config
    if args.modelo and args.modelo != 'aleatoria':
        d = os.path.dirname(args.modelo.replace('.zip', ''))
        for ruta in (os.path.join(d, 'config.json'),                   # p1_logs/<run>/modelo
                     os.path.join(os.path.dirname(d), 'config.json')):  # .../checkpoints/modelo_N_steps
            if os.path.exists(ruta):
                print('Configuración del modelo: ' + ruta)
                return ruta
    return 'p1_config.json'


def cargar_politica(args, env):
    """Devuelve (función obs -> acción, nombre). 'aleatoria' es la línea base."""
    if args.modelo in (None, 'aleatoria'):
        return (lambda obs: env.action_space.sample()), 'aleatoria'
    from stable_baselines3 import PPO, SAC
    ruta = args.modelo
    base = os.path.basename(ruta.replace('.zip', ''))
    nombre = os.path.basename(os.path.dirname(ruta)) if base == 'modelo' else base
    nombre = nombre or base
    texto = ruta.lower()
    Algoritmo = PPO if 'ppo' in texto else SAC if 'sac' in texto else \
        (SAC if args.algoritmo == 'sac' else PPO)
    modelo = Algoritmo.load(ruta, device='cpu')
    return (lambda obs: modelo.predict(obs, deterministic=True)[0]), nombre


def modo_evaluar(args):
    """Protocolo del anexo B.

    Cada episodio empieza con un reinicio de la escena y el seguidor
    parado, seguido de la acción de arranque predeterminada (B.2 la
    admite: el guía no arranca hasta que el seguidor se acerca); la
    política es determinista. Termina a los 120 s (completado) o en el
    primer paso con un fallo de B.3, calculado aquí con la verdad del
    terreno e independientemente de la recompensa. La configuración del
    guía (C1, C2 o C3) se fija a mano en el menú del simulador; aquí sólo
    se anota. Si el simulador se congela, el episodio se anota como
    'congelado' y hay que repetirlo.
    """
    from p1_protocolo import RegistroEvaluacion, leer_episodios, agrupar, imprimir_tabla
    cfg = cargar_config(config_del_modelo(args))
    p = cfg['protocolo']
    env = crear_env(args, modo='protocolo', config=cfg, pasos_max=10 ** 9)
    politica, nombre = cargar_politica(args, env)
    conf = args.configuracion
    if not args.sin_pausa:
        input('\nConfigura el guía en RoboboSim como {} (C1: crucero 10-10, C2: crucero '
              '10-40, C3: variable 10-40), pulsa Apply, vuelve aquí, pulsa Enter y deja '
              'la ventana del simulador en primer plano... '.format(conf))
    fecha = time.strftime('%Y%m%d-%H%M%S')
    rutas = []
    try:
        for ep in range(1, (args.episodios or 1) + 1):
            ruta = os.path.join(RUTA_EVAL, '{}_{}_{}_ep{}.csv'.format(nombre, conf, fecha, ep))
            reg = RegistroEvaluacion(ruta)
            rutas.append(ruta)
            obs, info = env.reset()
            paso, causa, t = 0, '', 0.0
            try:
                while not causa:
                    accion = politica(obs)
                    obs, r, term, trunc, info = env.step(accion)
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
        print()
        imprimir_tabla(agrupar(leer_episodios(rutas)))


# =====================================================================
# --ejecutar : la política sin fin (vídeos y robot real)
# =====================================================================

def modo_ejecutar(args):
    """Ejecuta la política determinista hasta Ctrl-C.

    En el simulador reinicia la escena tras cada fallo. Con
    --sin-simulador (robot real) no hay verdad del terreno: ni recompensa
    ni terminaciones, sólo la política sobre los sensores.
    """
    env = crear_env(args, modo='protocolo', config=config_del_modelo(args), pasos_max=10 ** 9)
    politica, nombre = cargar_politica(args, env)
    print('Ejecutando {} (Ctrl-C para parar)'.format(nombre))
    try:
        obs, _ = env.reset()
        while True:
            obs, r, term, trunc, info = env.step(politica(obs))
            if term or trunc:
                print('Fin: {} a los {:.1f} s'.format(info['causa'], info['t']))
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# =====================================================================

def main():
    p = argparse.ArgumentParser(description='Práctica 1 de RIA: seguimiento del vehículo guía.')
    modo = p.add_mutually_exclusive_group(required=True)
    for m in ['calibrar-movimiento', 'barrido-tilt', 'grabar-linea', 'caracterizar',
              'comprobar', 'aleatorio', 'entrenar', 'evaluar', 'ejecutar']:
        modo.add_argument('--' + m, action='store_true')
    p.add_argument('--algoritmo', choices=['sac', 'ppo'], default='sac')
    p.add_argument('--pasos', type=int, default=30_000, help='pasos de entrenamiento')
    p.add_argument('--episodios', type=int, default=None,
                   help='--aleatorio: 10 por defecto; --evaluar: 1 por defecto')
    p.add_argument('--pasos-max', type=int, default=None, help='por defecto, el de p1_config.json')
    p.add_argument('--semilla', type=int, default=0)
    p.add_argument('--etiqueta', default='', help='p. ej. C3: entra en el nombre de los registros')
    p.add_argument('--continuar', default=None, help='modelo .zip desde el que seguir entrenando')
    p.add_argument('--cada', type=int, default=2000, help='pasos entre copias del modelo')
    p.add_argument('--sin-parada', action='store_true',
                   help='--entrenar sin parada temprana (p. ej., semillas con el mismo presupuesto)')
    p.add_argument('--modelo', default=None, help='modelo a evaluar/ejecutar, o "aleatoria"')
    p.add_argument('--configuracion', default='C3', choices=['C1', 'C2', 'C3', 'otra'],
                   help='configuración del guía en la evaluación (sólo se anota)')
    p.add_argument('--sin-pausa', action='store_true', help='--evaluar sin esperar a Enter')
    p.add_argument('--config', default=None,
                   help='por defecto p1_config.json; con --evaluar/--ejecutar, la guardada con el modelo')
    p.add_argument('--sin-simulador', action='store_true')
    p.add_argument('--detalle', action='store_true', help='una línea por paso')
    p.add_argument('--no-guardar', action='store_true', help='--calibrar-movimiento sin tocar la config')
    p.add_argument('--guardar', action='store_true',
                   help='--caracterizar / --barrido-tilt: escribe el resultado en la config')
    p.add_argument('--tilts', default='60,70,80,90,100', help='--barrido-tilt: grados')
    p.add_argument('--distancias', default='0.15,0.20,0.30,0.45,0.60,0.75,0.90',
                   help='--barrido-tilt: m por detrás de la pelota')
    p.add_argument('--distancias-car', default='1.2,1.0,0.9,0.8,0.7,0.6,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15',
                   help='--caracterizar: m por detrás de la pelota')
    p.add_argument('--giro', type=float, default=15.0,
                   help='--caracterizar: desvío del rumbo en grados para medir el campo de visión')
    args = p.parse_args()

    os.makedirs(RUTA_LOGS, exist_ok=True)
    if args.config is None and not (args.evaluar or args.ejecutar):
        args.config = 'p1_config.json'
    if args.calibrar_movimiento:
        modo_calibrar_movimiento(args)
    elif args.barrido_tilt:
        modo_barrido_tilt(args)
    elif args.grabar_linea:
        modo_grabar_linea(args)
    elif args.caracterizar:
        modo_caracterizar(args)
    elif args.comprobar:
        modo_comprobar(args)
    elif args.aleatorio:
        modo_aleatorio(args)
    elif args.entrenar:
        modo_entrenar(args)
    elif args.evaluar:
        modo_evaluar(args)
    elif args.ejecutar:
        modo_ejecutar(args)


if __name__ == '__main__':
    main()
