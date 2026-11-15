#!/usr/bin/env python3
"""
Práctica 1 — Entorno Gymnasium del seguidor en "Follow City".

Tarea: el Robobo seguidor circula detrás del vehículo guía, que lleva una
pelota verde en su parte trasera, manteniendo la separación en la banda
objetivo, sin chocar, sin perderlo de vista y, si es posible, sin salirse
de su carril. El guía cambia de velocidad (frena en las curvas y, en modo
variable, fluctúa también en las rectas): el seguidor lo detecta porque la
pelota cambia de tamaño en la imagen y ajusta su propia velocidad.

Misma estructura que el ejemplo de la pared (p1_robobo_env.py): el
algoritmo sólo ve reset() y step(); todo lo demás lo esconde el entorno.

--------------------------------------------------------------------
CANALES
--------------------------------------------------------------------

  ROS 2, nodo puente del seguidor   percepción (blobs, IR) y actuación
  (base/ y smartphone/)             (cmd_vel, tilt). Es lo ÚNICO que entra
                                    en la observación, porque es lo único
                                    que existe en el robot real.
  Conexión directa con RoboboSim    verdad del terreno (posiciones del
  (p1_simulador.py)                 seguidor, del guía y de su pelota),
                                    reinicio de la escena y colocación del
                                    seguidor. Sólo para recompensa,
                                    terminaciones, métricas y reinicios.
                                    (Por qué no el módulo sim del puente:
                                    ver p1_simulador.py.)

--------------------------------------------------------------------
REQUISITOS
--------------------------------------------------------------------

  RoboboSim con "Follow City" y SU VENTANA EN PRIMER PLANO: el simulador
  congela la física cuando pierde el foco (ver congelado()).

  Puente del seguidor (sólo el módulo de blobs):
    source /root/robobo_ws/install/setup.bash
    ros2 run robobo_ros2 robobo_container --ros-args \\
         -p ip:=host.docker.internal -p modules:="['blob']"
  (o  bash p1_puentes.sh start)

Los parámetros de calibración están en p1_config.json.
"""

import math
import os
import time
from collections import deque

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

from robobo_ros2_interfaces.msg import BlobArray

try:
    from robobo_ros2_interfaces.srv import MoveTilt
except ImportError:
    MoveTilt = None

from p1_comun import (cargar_config, ConvencionRumbo, LineaCentral, ModeloCamara,
                      posicion_pelota, desviacion, angulo_hacia, envolver,
                      proximidad_ir, fallo_protocolo)
from p1_recompensa import Recompensa

AVISO_CONGELADO = (
    '*** El robot no se mueve aunque se le manda girar. Dos causas posibles:\n'
    '    1. La ventana de RoboboSim no está en primer plano (sin foco congela la física):\n'
    '       haz clic en ella.\n'
    '    2. Has cerrado/reabierto RoboboSim o recargado el mundo después de lanzar el\n'
    '       puente: el puente pierde la conexión y no se reconecta solo. Reinícialo\n'
    '       (Ctrl-C en su terminal y otra vez ros2 run ... modules:="[\'blob\']").\n'
    '    Esperando... ***')

ESPERA_MAX_S = 3.0         # espera máxima de un mensaje de blobs (se publican a 5 Hz)
ESPERA_FRAME_S = 0.6       # espera máxima a que cambie el blob (en RoboboSim cambia a ~1,8 Hz)

# Nombre de cada componente de la observación, en orden.
NOMBRES_OBS = ['visible', 'x', 'y', 'distancia', 'vel_x', 'vel_distancia',
               'vel_guia', 'sin_pelota', 'ir_frontal', 'v_anterior', 'w_anterior']


class RoboboSeguidorEnv(gym.Env):
    """Seguimiento del vehículo guía en Follow City.

    Observación (Box, float32, dimensión 11). Sólo sensores del seguidor.

      [0] visible        1 si el detector ve la pelota verde, 0 si no.
      [1] x              posición horizontal del blob, (x-50)/50 -> [-1, 1].
                         Si no se ve: el último valor visto, que indica por
                         dónde se fue.
      [2] y              posición vertical del blob, igual. Cambia con la
                         distancia y con el cabeceo del robot.
      [3] distancia      distancia estimada a la pelota con el modelo de
                         la cámara, d = k·size^(-p) (p1_config.json),
                         dividida por la distancia de "guía perdido"
                         (0,90 m) -> [0, 1.5]. Si no se ve: la última.
      [4] vel_x          velocidad de la pelota en la imagen (x_norm/s),
                         dividida por la que produciría girar a w_max ->
                         [-1, 1]. Separa "el guía entra en una curva" de
                         "yo estoy girando".
      [5] vel_distancia  derivada de [3] en m/s dividida por v_max ->
                         [-1, 1]. Positiva: el guía se aleja.
      [6] vel_guia       estimación de la velocidad del guía:
                             v_guia ≈ v_propia + d(distancia)/dt
                         (la propia es la ordenada en los mismos pasos, en
                         m/s reales), dividida por v_max -> [-1, 1]. Es la
                         componente que permite detectar que el guía
                         acelera o frena y adaptar la velocidad a él. Si no
                         se ve, la última estimación.
      [7] sin_pelota     tiempo sin ver la pelota / 3 s -> [0, 1].
      [8] ir_frontal     proximidad logarítmica del IR frontal más alto ->
                         [0, 1]: cubre la zona muy cercana, donde la
                         pelota puede salirse de la imagen.
      [9] v_anterior     acción lineal del paso anterior, [-1, 1].
     [10] w_anterior     acción angular del paso anterior, [-1, 1].

    Las velocidades [4]-[6] se calculan entre la lectura actual y la de
    hace 'ventana_derivada_pasos' pasos, sólo si la pelota se ha visto en
    todas ellas; si no, valen 0.

    Acción (Box, float32, dimensión 2), reescalada dentro del entorno a
    velocidades REALES:
      [0] v en [v_min, v_max] m/s     [1] w en [-w_max, w_max] rad/s
    Se publica como Twist en base/cmd_vel dividida por factor_v y factor_w
    (velocidad real / pedida, medidas con --calibrar-movimiento): en el
    simulador el robot va ~1,9 veces más rápido de lo que pide cmd_vel.

    Recompensa: p1_recompensa.py.

    Modos:
      'entrenamiento'  terminación por los fallos del protocolo y además
                       por adelantar al guía o salirse de la calzada (o
                       del carril, si se activa en p1_config.json).
                       Reinicio según reinicio.entrenamiento (mixto).
      'protocolo'      sólo los fallos del anexo B.3, y reinicio de la
                       escena; lo usa el evaluador.
    En los dos, si el simulador se congela el episodio se trunca con la
    causa 'congelado' y el entorno espera a que vuelva a correr.
    """

    metadata = {'render_modes': []}

    def __init__(self, config='p1_config.json', modo='entrenamiento',
                 usar_simulador=True, pasos_max=None, verbose=False):
        super().__init__()
        if modo not in ('entrenamiento', 'protocolo'):
            raise ValueError('modo desconocido: ' + str(modo))
        self.cfg = cargar_config(config) if isinstance(config, str) else config
        c = self.cfg
        self.modo = modo
        self.usar_simulador = usar_simulador
        self.pasos_max = pasos_max or c['pasos_max']
        self.verbose = verbose

        a = c['accion']
        self.v_min, self.v_max, self.w_max = a['v_min'], a['v_max'], a['w_max']
        self.factor_v = a.get('factor_v', 1.0)
        self.factor_w = a.get('factor_w', 1.0)
        self.paso_s = c['paso_s']
        self.keepalive_s = c['keepalive_s']
        self.p = c['protocolo']

        cam = c['camara']
        self.color = cam['blob_color']
        self.size_min = cam['blob_size_min']
        self.caducidad = cam['blob_caducidad_s']
        self.camara = ModeloCamara(c)
        self.rumbo = ConvencionRumbo(c['rumbo']['signo'], c['rumbo']['offset_deg'])
        self.recompensa = Recompensa(c)
        self.ventana = max(1, int(c['observacion']['ventana_derivada_pasos']))

        # Escala física de [4]: girando a w_max, un objeto cerca del centro
        # de la imagen se desplaza a w_max/tan(fov/2) unidades de x_norm por s.
        self.escala_vel_x = self.w_max / self.camara.tan_semi

        ns = '/robobo/robot_{}'.format(c['robot_seguidor'])

        # -------------------------------------------------- ROS 2
        if not rclpy.ok():
            rclpy.init()
        self.nodo = Node('p1_seguidor_env')

        self.pub_vel = self.nodo.create_publisher(Twist, ns + '/base/cmd_vel', 10)
        self._t_pub = 0.0                    # instante de la última orden publicada
        self._orden = (0.0, 0.0)             # última orden publicada (unidades de cmd_vel)

        # Blobs: contador de mensajes (para exigir una lectura posterior a
        # la acción) y pelota verde del último mensaje.
        self._n_blob = 0
        self._blob = None                    # (x, y, size, frame_timestamp) o None
        self._blob_fts = None                # último frame_timestamp distinto
        self._t_fts_cambio = 0.0             # instante local en que cambió
        self._fts_invalido = None            # frame anterior al reinicio
        self.nodo.create_subscription(
            BlobArray, ns + '/smartphone/color_blobs', self._cb_blob, 10)

        self._ir = None
        self.nodo.create_subscription(
            Int32MultiArray, ns + '/base/ir', self._cb_ir, 10)

        self.cli_tilt = None
        if MoveTilt is not None:
            self.cli_tilt = self.nodo.create_client(MoveTilt, ns + '/base/move_tilt')

        # -------------------------------------------------- simulador
        self.cliente = None
        self.linea = None
        self._pos_seg = self._pos_guia = self._pos_pelota = None
        self._firma, self._t_cambio = None, time.time()      # para detectar el congelado
        self._ref_dir, self._dir_guia = None, None           # dirección de avance del guía
        if usar_simulador:
            from p1_simulador import ClienteSim
            self.cliente = ClienteSim(c.get('ip_simulador', 'host.docker.internal'),
                                      c['robot_seguidor'], c['objeto_guia'],
                                      c.get('objeto_pelota'))
            ruta = c['carril'].get('archivo_linea_central', '')
            if ruta and os.path.exists(ruta):
                self.linea = LineaCentral(np.load(ruta))
            elif modo == 'entrenamiento':
                print('AVISO: no existe "{}": no se comprueba ni la calzada ni el '
                      'carril y los episodios empiezan siempre reiniciando la escena. '
                      'Grábala con  python3 p1_seguidor.py --grabar-linea'.format(ruta))

        # -------------------------------------------------- espacios
        bajo = np.array([0, -1, -1, 0.0, -1, -1, -1, 0, 0, -1, -1], dtype=np.float32)
        alto = np.array([1, 1, 1, 1.5, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(bajo, alto, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # -------------------------------------------------- estado
        self._reiniciar_estado()

        # Confirmar que el puente responde antes de entrenar
        self._esperar_blob_nuevo(inicial=True)
        self.colocar_tilt()
        if usar_simulador:
            self._medidas()

    def _reiniciar_estado(self):
        ahora = time.time()
        self.pasos = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)
        self._v_hist = deque(maxlen=self.ventana)        # v real ordenada en los últimos pasos
        self._hist = deque(maxlen=self.ventana + 1)      # (t, d, x) de lecturas visibles seguidas
        self.ultimo_x = 0.0
        self.ultimo_y = 0.0
        self.ultima_d = self.p['perdido_m']              # sin verla aún: "puede estar lejos"
        self._v_guia = 0.0
        self.t_visto = ahora
        self.t_inicio = ahora
        self._t_cambio = ahora
        self._d_ant = None
        self.tipo_reinicio = ''

    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------

    def _cb_blob(self, msg):
        verde = None
        for b in msg.blobs:
            if b.color == self.color:
                verde = (b.x, b.y, b.size, b.frame_timestamp)
        self._blob = verde
        if verde is not None and verde[3] != self._blob_fts:
            self._blob_fts = verde[3]
            self._t_fts_cambio = time.time()
        self._n_blob += 1

    def _cb_ir(self, msg):
        self._ir = list(msg.data)

    def orden_cmd(self, v, w):
        """Velocidades reales (m/s, rad/s) -> lo que hay que pedir a cmd_vel."""
        return v / self.factor_v, w / self.factor_w

    def accion_normalizada(self, v, w):
        """Velocidades reales -> acción en [-1, 1] (inversa del reescalado)."""
        return np.array([2.0 * (v - self.v_min) / (self.v_max - self.v_min) - 1.0,
                         w / self.w_max], dtype=np.float32)

    def _publicar_velocidad(self, v_cmd, w_cmd):
        """Publica una orden en unidades de cmd_vel (sin corregir)."""
        msg = Twist()
        msg.linear.x = float(v_cmd)
        msg.angular.z = float(w_cmd)
        self.pub_vel.publish(msg)
        self._t_pub = time.time()
        self._orden = (float(v_cmd), float(w_cmd))

    def _mantener_orden(self):
        """Re-publica la última orden cada keepalive_s.

        El puente para las ruedas si pasan 0,5 s sin recibir cmd_vel. Un
        paso dura paso_s más la espera de la lectura (hasta ~0,45 s), así
        que sin re-publicar el robot se pararía a ratos dentro del paso.
        """
        if time.time() - self._t_pub >= self.keepalive_s:
            self._publicar_velocidad(*self._orden)

    def _aplicar(self, v_cmd, w_cmd, segundos):
        """Publica la orden y la mantiene 'segundos', procesando los mensajes.

        A diferencia del ejemplo de la pared, aquí sí se hace spin durante
        la aplicación: hay dos suscripciones y procesarlas sobre la marcha
        evita que se acumulen en las colas. Lo que garantiza que la
        observación es posterior a la acción es _esperar_blob_nuevo(), que
        exige un mensaje de blobs procesado DESPUÉS de este periodo.
        """
        self._publicar_velocidad(v_cmd, w_cmd)
        self._esperar(segundos)

    def _esperar(self, segundos):
        """Deja pasar el tiempo procesando mensajes y manteniendo la orden."""
        t0 = time.time()
        while True:
            restante = segundos - (time.time() - t0)
            if restante <= 0:
                return
            rclpy.spin_once(self.nodo, timeout_sec=min(0.02, restante))
            self._mantener_orden()

    def _esperar_blob_nuevo(self, blob_antes=None, inicial=False):
        """Bloquea hasta procesar un mensaje de blobs POSTERIOR a este instante.

        Es el requisito técnico del enunciado: si step() devolviera la
        última lectura guardada, la observación sería anterior a la acción
        y el algoritmo recibiría pares (observación, acción) desplazados
        un paso, sin ningún mensaje de error.

        Un mensaje posterior no basta: el puente publica a 5 Hz el último
        blob que conoce aunque no haya imagen nueva, y en RoboboSim el
        detector va más despacio (el mismo valor se repite hasta 0,6 s) y
        frame_timestamp vale siempre 0. Por eso, si se da blob_antes (el
        blob que había al publicar la acción) y la pelota se ve, se espera
        además, como mucho ESPERA_FRAME_S, a que cambie: o a que cambie el
        frame_timestamp, si el detector lo informa (robot real), o a que
        cambien x, y o el tamaño. Con el guía siempre en marcha, un blob
        idéntico es, casi seguro, la misma imagen. Mientras tanto se sigue
        manteniendo la orden.
        """
        objetivo = self._n_blob + 1
        t0 = time.time()
        while self._n_blob < objetivo:
            rclpy.spin_once(self.nodo, timeout_sec=0.02)
            self._mantener_orden()
            if time.time() - t0 > ESPERA_MAX_S:
                raise RuntimeError(
                    'No llegan mensajes de smartphone/color_blobs. ¿Está el puente '
                    'del seguidor lanzado con el módulo blob?' if inicial else
                    'Se ha interrumpido el flujo de blobs durante el episodio.')
        if blob_antes is None or not self._visible():
            return

        def igual():
            b = self._blob
            if b[3] > 0:                       # el detector informa del frame
                return b[3] == blob_antes[3]
            return b[:3] == blob_antes[:3]

        t1 = time.time()
        while igual() and time.time() - t1 < ESPERA_FRAME_S:
            rclpy.spin_once(self.nodo, timeout_sec=0.02)
            self._mantener_orden()

    def _detener(self):
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def colocar_tilt(self, angulo=None, velocidad=50.0):
        """Fija la inclinación de la cámara (servicio base/move_tilt)."""
        angulo = self.cfg['camara'].get('tilt_deg') if angulo is None else angulo
        if angulo is None or self.cli_tilt is None or \
                not self.cli_tilt.wait_for_service(timeout_sec=1.0):
            return
        req = MoveTilt.Request()
        req.angle = float(angulo)
        req.speed = float(velocidad)
        futuro = self.cli_tilt.call_async(req)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)

    # -----------------------------------------------------------------
    # Simulador: congelado, colocación y arranque
    # -----------------------------------------------------------------

    def congelado(self):
        """¿Lleva la escena más de congelado_s sin cambiar ni un poco?

        RoboboSim (Unity) congela la física cuando su ventana pierde el
        foco, aunque siga recibiendo órdenes y enviando posiciones. Se da
        por congelado si en ese tiempo no se han movido ni el seguidor (y
        se le está mandando moverse) ni el guía. No se puede mirar sólo al
        guía: tras reiniciar la escena está quieto a propósito hasta que
        el seguidor se acerca.
        """
        mandando = abs(self._orden[0]) > 0.005 or abs(self._orden[1]) > 0.01
        return mandando and time.time() - self._t_cambio > self.cfg.get('congelado_s', 1.5)

    def _girar_hasta_ver_cambio(self, w_cmd, maximo_s):
        """Gira en el sitio hasta que cambie la rotación medida (o maximo_s).

        Devuelve el tiempo que ha girado, o None si no ha cambiado nada. El
        simulador envía la posición del robot a ~4 Hz, así que hay que dar
        margen: con 0,25 s fijos la prueba fallaba a veces sin motivo.
        """
        yaw0 = self.cliente.seguidor()[2]
        self._publicar_velocidad(0.0, w_cmd)
        t0 = time.time()
        while time.time() - t0 < maximo_s:
            self._esperar(0.03)
            if abs(math.degrees(envolver(math.radians(self.cliente.seguidor()[2] - yaw0)))) > 0.3:
                return time.time() - t0
        return None

    def esperar_simulador_activo(self):
        """Bloquea hasta comprobar que la física avanza.

        La prueba es un giro en el sitio a un lado (hasta que la rotación
        medida cambie, como mucho 1 s) y otro tanto de vuelta: el seguidor
        acaba casi en el mismo rumbo.
        """
        avisado = False
        while True:
            t = self._girar_hasta_ver_cambio(0.3, 1.0)
            if t is not None:
                self._aplicar(0.0, -0.3, t)
                self._detener()
                if avisado:
                    print('*** El simulador vuelve a correr. ***')
                self._t_cambio = time.time()
                return
            self._detener()
            if not avisado:
                print(AVISO_CONGELADO)
                avisado = True
            self._esperar(0.5)

    def esperar_blob_fresco(self, maximo_s=1.0):
        """Espera a que el blob cambie (una imagen nueva) o maximo_s."""
        antes = self._blob
        t0 = time.time()
        while time.time() - t0 < maximo_s:
            self._esperar_blob_nuevo()
            if self._blob != antes:
                return True
        return False

    def colocar_seguidor_detras(self, distancia, en_linea=True, ruido=False, giro_deg=0.0):
        """Coloca al seguidor 'distancia' m por detrás de la pelota, mirándola.

        Con la línea central, sobre ella (a esa distancia a lo largo de la
        línea) y mirando a la pelota: vale en rectas y curvas. (Mirando en
        la dirección de la línea, en las curvas la pelota podía quedar fuera
        del campo de visión y el episodio empezaba con el guía ya perdido.)
        Sin ella, sobre la recta que une su posición actual con la pelota
        (tras un reinicio salen alineados en la recta). Con ruido se añade
        un error de rumbo y lateral aleatorio (reinicio.ruido_*); giro_deg
        desvía el rumbo esa cantidad (caracterización del campo de visión).
        """
        self._medidas()
        pelota = self._pos_pelota
        if en_linea and self.linea is not None:
            s_p, _ = self.linea.proyectar(pelota)
            punto, tang = self.linea.punto(s_p - distancia)
            if ruido:
                r = self.cfg['reinicio']
                normal = np.array([tang[1], -tang[0]])
                punto = punto + normal * self.np_random.uniform(-1, 1) * r['ruido_lateral_m']
            rumbo = math.atan2(pelota[0] - punto[0], pelota[1] - punto[1])
            if ruido:
                rumbo += math.radians(self.np_random.uniform(-1, 1) * r['ruido_rumbo_deg'])
        else:
            f = np.array(self._pos_seg[:2])
            u = (pelota - f) / max(np.linalg.norm(pelota - f), 1e-6)
            punto = pelota - distancia * u
            rumbo = math.atan2(u[0], u[1])
        rumbo += math.radians(giro_deg)
        yaw = math.degrees(envolver(rumbo - self.rumbo.offset)) * self.rumbo.signo
        self._detener()
        # La posición llega a ~4 Hz y a veces la colocación tarda en verse:
        # se espera a comprobarla (máx. 1 s) y, si no, se repite una vez.
        for _ in range(2):
            self.cliente.colocar_seguidor(punto[0], punto[1], yaw)
            t0 = time.time()
            while time.time() - t0 < 1.0:
                self._esperar(0.05)
                if np.linalg.norm(np.array(self.cliente.seguidor()[:2]) - punto) < 0.03:
                    self._esperar(0.1)
                    return
        print('AVISO: no se ha podido comprobar la colocación del seguidor.')

    def _arranque(self):
        """Acción de arranque predeterminada tras reiniciar la escena.

        En Follow City el guía no arranca hasta que el seguidor está a
        menos de ~0,35-0,40 m de su pelota, y el seguidor sale a 0,645 m.
        El anexo B.2 admite acciones de arranque predeterminadas: avanzar
        recto hasta que el blob estime arranque_distancia_m (sólo sensores
        del robot, igual que en el robot real) o, si no se ve la pelota,
        durante arranque_t_max_s. Se entrega el control en marcha.
        """
        r = self.cfg['reinicio']
        v = r['arranque_v']
        self._publicar_velocidad(*self.orden_cmd(v, 0.0))
        t0 = time.time()
        while time.time() - t0 < r['arranque_t_max_s']:
            self._esperar_blob_nuevo()
            if self._visible() and self.camara.distancia(self._blob[2]) <= r['arranque_distancia_m']:
                break
        self.ultima_accion = self.accion_normalizada(v, 0.0)
        self._v_hist.append(v)

    # -----------------------------------------------------------------
    # Observación
    # -----------------------------------------------------------------

    def _visible(self):
        """¿Se ve la pelota en la última lectura?

        Además de exigir tamaño > 0, si el simulador informa del instante
        del frame se descarta una lectura cuyo frame lleva más de
        blob_caducidad_s sin cambiar (el puente reenvía el último blob
        conocido aunque la pelota haya desaparecido) o que es la que había
        antes del reinicio de la escena.
        """
        b = self._blob
        if b is None or b[2] < self.size_min:
            return False
        if b[3] > 0:
            if b[3] == self._fts_invalido:
                return False
            return time.time() - self._t_fts_cambio < self.caducidad
        return True

    def _ir_frontal(self):
        if self._ir is None:
            return 0.0
        sat = self.cfg['ir']['saturacion']
        return max(proximidad_ir(self._ir[i], sat) for i in self.cfg['ir']['indices'])

    def _observacion(self):
        ahora = time.time()
        visible = self._visible()
        vel_d = vel_x = 0.0
        if visible:
            x, y, size, _ = self._blob
            xn = self.camara.x_normalizada(x)
            yn = self.camara.y_normalizada(y)
            d = self.camara.distancia(size)
            self._hist.append((ahora, d, xn))
            if len(self._hist) >= 2:
                t0, d0, x0 = self._hist[0]
                dt = max(ahora - t0, 1e-3)
                vel_d = (d - d0) / dt
                vel_x = (xn - x0) / dt
                v_propia = float(np.mean(self._v_hist)) if self._v_hist else 0.0
                self._v_guia = v_propia + vel_d
            self.ultimo_x, self.ultimo_y, self.ultima_d = xn, yn, d
            self.t_visto = ahora
        else:
            self._hist.clear()
        t_sin = ahora - self.t_visto

        obs = np.array([
            float(visible),
            self.ultimo_x,
            self.ultimo_y,
            np.clip(self.ultima_d / self.p['perdido_m'], 0.0, 1.5),
            np.clip(vel_x / self.escala_vel_x, -1.0, 1.0),
            np.clip(vel_d / self.v_max, -1.0, 1.0),
            np.clip(self._v_guia / self.v_max, -1.0, 1.0),
            min(t_sin / self.p['sin_pelota_s'], 1.0),
            self._ir_frontal(),
            self.ultima_accion[0],
            self.ultima_accion[1],
        ], dtype=np.float32)
        percepcion = {'visible': visible, 'x': self.ultimo_x, 'd': self.ultima_d,
                      'vel_d': vel_d, 'v_guia': self._v_guia, 't_sin': t_sin}
        return obs, percepcion

    # -----------------------------------------------------------------
    # Verdad del terreno (sólo recompensa, terminación y métricas)
    # -----------------------------------------------------------------

    def _medidas(self):
        """Magnitudes del anexo B a partir de las posiciones del simulador.

        La separación es la distancia del centro del seguidor a la pelota
        (el objeto pelota de la escena). Con la línea central grabada se
        añaden el error lateral, si el seguidor está en su carril y en la
        calzada, y si adelanta al guía. De paso actualiza la "firma" de la
        escena con la que se detecta el simulador congelado.
        """
        car = self.cfg['carril']
        sx, sz, syaw, _ = self.cliente.seguidor()
        g = self.cliente.guia()
        firma = np.array([sx, sz, syaw, g[0], g[1]])
        if self._firma is None or np.max(np.abs(firma - self._firma)) > 1e-5:
            self._t_cambio = time.time()
        self._firma = firma
        if self._ref_dir is None:
            self._ref_dir = g
        elif np.linalg.norm(g - self._ref_dir) > 0.01:
            self._dir_guia = angulo_hacia(self._ref_dir, g)
            self._ref_dir = g
        pelota = self.cliente.pelota()
        if pelota is None:
            pelota = posicion_pelota(g, self._dir_guia or 0.0, self.p['offset_pelota_m'])
        self._pos_seg, self._pos_guia, self._pos_pelota = (sx, sz, syaw), g, pelota

        f = np.array([sx, sz])
        m = {'separacion': float(np.linalg.norm(pelota - f)),
             'desviacion': desviacion(f, self.rumbo(syaw), pelota),
             'pos_seguidor': f, 'pos_guia': g, 'pos_pelota': pelota,
             'error_lateral': float('nan'), 'en_carril': None,
             'en_carril_estricto': None, 'en_calzada': True, 'adelanta': False}
        if self.linea is not None:
            s_p, _ = self.linea.proyectar(pelota)
            s_f, e = self.linea.proyectar(f)
            m.update(error_lateral=e,
                     en_carril=abs(e) <= car['ancho_carril_m'] / 2.0,
                     en_carril_estricto=abs(e) <= (car['ancho_carril_m'] - car['ancho_robot_m']) / 2.0,
                     en_calzada=-car['calzada_exterior_m'] <= e <= car['calzada_interior_m'],
                     adelanta=self.linea.separacion(s_p, s_f) < 0)
        return m

    def _causa_fallo(self, m, t_sin):
        causa = fallo_protocolo(m['separacion'], t_sin, self.p)
        if causa or self.modo == 'protocolo':
            return causa
        car = self.cfg['carril']
        if m['adelanta']:
            return 'adelanta'
        if car['terminar_fuera_calzada'] and not m['en_calzada']:
            return 'fuera_calzada'
        if car['terminar_fuera_carril'] and m['en_carril'] is False:
            return 'fuera_carril'
        return ''

    def _info(self, m):
        """Diccionario info con las magnitudes de la verdad del terreno."""
        banda = self.p['banda_m']
        desv_deg = math.degrees(m['desviacion'])
        return {
            'separacion': m['separacion'],
            'desviacion_deg': desv_deg,
            'error_lateral': m['error_lateral'],
            'en_banda': banda[0] <= m['separacion'] <= banda[1],
            'desviacion_ok': abs(desv_deg) < self.p['desviacion_ok_deg'],
            'en_carril': m['en_carril'],
            'en_carril_estricto': m['en_carril_estricto'],
            'en_calzada': m['en_calzada'],
            'pos_seguidor': tuple(float(v) for v in m['pos_seguidor']),
            'pos_guia': tuple(float(v) for v in m['pos_guia']),
            'pos_pelota': tuple(float(v) for v in m['pos_pelota']),
        }

    # -----------------------------------------------------------------
    # Interfaz de Gymnasium
    # -----------------------------------------------------------------

    def _elegir_reinicio(self):
        if self.modo == 'protocolo' or self.linea is None:
            return 'escena'
        tipo = self.cfg['reinicio']['entrenamiento']
        if tipo == 'mixto':
            return 'escena' if self.np_random.random() < self.cfg['reinicio']['prob_escena'] \
                else 'colocar'
        return tipo

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._detener()
        tipo = (options or {}).get('reinicio') or ('' if self.cliente is None
                                                   else self._elegir_reinicio())

        if tipo == 'escena':
            # Ambos robots vuelven a donde los coloca el escenario y el
            # seguidor parte parado (anexo B.2); luego, acción de arranque.
            self.cliente.reiniciar()
            self._esperar(self.cfg['espera_reset_s'])
            self.esperar_simulador_activo()
            self.colocar_tilt()
        elif tipo == 'colocar':
            # Sin reiniciar la escena: el guía sigue circulando y el
            # seguidor aparece detrás de él, a una distancia aleatoria de
            # la banda, en cualquier punto del recorrido.
            # (La inclinación de la cámara no cambia: no se reinicia la escena.)
            self.esperar_simulador_activo()
            lo, hi = self.cfg['reinicio']['separacion_colocar_m']
            self.colocar_seguidor_detras(self.np_random.uniform(lo, hi), ruido=True)
        else:
            input('\n[reset] Coloca el seguidor detrás del guía y pulsa Enter... ')
            self.colocar_tilt()

        self._reiniciar_estado()
        self.tipo_reinicio = tipo
        # El blob que hubiera antes del reinicio no vale como observación.
        self._fts_invalido = self._blob_fts
        self._esperar_blob_nuevo()

        # La cámara puede tardar un poco en ver la pelota tras el reinicio.
        t0 = time.time()
        while not self._visible() and time.time() - t0 < self.cfg['espera_pelota_s']:
            self._esperar_blob_nuevo()
        if tipo == 'escena':
            self._arranque()

        self.t_inicio = time.time()
        self.t_visto = self.t_inicio
        obs, perc = self._observacion()
        info = {'visible': perc['visible'], 'reinicio': tipo}
        if self.usar_simulador:
            m = self._medidas()
            self._d_ant = m['separacion']
            self._t_cambio = time.time()
            info.update(self._info(m))
        return obs, info

    def step(self, accion):
        a = np.clip(np.asarray(accion, dtype=np.float32).reshape(2), -1.0, 1.0)
        v = self.v_min + (float(a[0]) + 1.0) * 0.5 * (self.v_max - self.v_min)
        w = float(a[1]) * self.w_max
        blob_antes = self._blob

        # 1-2. Publicar la orden y mantenerla durante paso_s
        self._aplicar(*self.orden_cmd(v, w), self.paso_s)

        # 3. Esperar una lectura de blobs posterior a la aplicación
        self._esperar_blob_nuevo(blob_antes=blob_antes)

        # 4. Observación (incluye la acción que se acaba de aplicar)
        self.pasos += 1
        self.ultima_accion = a
        self._v_hist.append(v)
        obs, perc = self._observacion()

        # 5. Recompensa y terminación
        info = {'visible': perc['visible'], 'x': perc['x'], 'd_estimada': perc['d'],
                'v_guia_estimada': perc['v_guia'], 't_sin_pelota': perc['t_sin'],
                'v': v, 'w': w, 't': time.time() - self.t_inicio, 'causa': '',
                'reinicio': self.tipo_reinicio}
        terminated = truncated = False
        recompensa = 0.0
        if self.usar_simulador:
            m = self._medidas()
            info.update(self._info(m))
            if self.congelado():
                # El simulador se ha parado (ventana sin foco): el paso no
                # vale. Se trunca el episodio sin recompensa ni castigo y se
                # espera a que vuelva a correr; el reset siguiente recoloca.
                info['causa'] = 'congelado'
                truncated = True
                self.esperar_simulador_activo()
            else:
                causa = self._causa_fallo(m, perc['t_sin'])
                terminated = causa != ''
                theta = self.camara.angulo(perc['x']) if perc['visible'] else None
                recompensa, desglose = self.recompensa(
                    self._d_ant, m['separacion'], theta, perc['visible'],
                    m['error_lateral'], terminated)
                self._d_ant = m['separacion']
                info.update(desglose)
                info['causa'] = causa

        if not terminated and not truncated and self.pasos >= self.pasos_max:
            truncated = True
            info['causa'] = 'tiempo'
        if terminated or truncated:
            self._detener()

        if self.verbose:
            print('paso {:3d}  vis={:d} x={:+.2f} d^={:.2f} vg^={:+.3f}  sep={:.3f} '
                  'desv={:+5.1f}  v={:+.3f} w={:+.2f}  r={:+.3f}  {}'.format(
                      self.pasos, int(perc['visible']), perc['x'], perc['d'], perc['v_guia'],
                      info.get('separacion', float('nan')),
                      info.get('desviacion_deg', float('nan')), v, w, recompensa,
                      info['causa']))

        return obs, recompensa, terminated, truncated, info

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        try:
            if self.cliente is not None:
                self.cliente.cerrar()
        except Exception:
            pass
        try:
            self.nodo.destroy_node()
        except Exception:
            pass
