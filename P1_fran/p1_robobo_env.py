"""
p1_robobo_env.py — Entorno Gymnasium para el Robobo en RoboboSim (Follow City).

Tarea: Controlar mediante aprendizaje por refuerzo a un Robobo seguidor para
que siga a un vehículo guía equipado con una pelota verde en su parte trasera.
"""

import time
import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist
from robobo_ros2_interfaces.msg import BlobArray

import config as cfg

# Interfaces propias del puente. Las define el módulo sim, que es
# reciente: en imágenes del contenedor anteriores a él no existen, y por
# eso la importación no puede ser obligatoria. Sin ellas el entorno
# funciona igual, pero no puede reiniciar la escena.
try:
    from robobo_ros2_interfaces.msg import RobotLocation
    from robobo_ros2_interfaces.srv import ResetSimulation
    HAY_INTERFACES_SIM = True
except ImportError:
    RobotLocation = ResetSimulation = None
    HAY_INTERFACES_SIM = False


# Espacios de nombres ROS 2
NS_BASE = '/robobo/robot_0/base'
NS_SMARTPHONE = '/robobo/robot_0/smartphone'
NS_SIM = '/robobo/robot_0/sim'

IR_FRONTALES = [0, 1, 2, 3, 4]
IR_SATURACION = 1000.0


class RoboboFollowEnv(gym.Env):
    """Entorno Gymnasium para el seguimiento del robot guía en Follow City."""

    metadata = {'render_modes': []}

    def __init__(self, pasos_max=300, usar_simulador=True, verbose=False):
        super().__init__()

        self.pasos_max = pasos_max
        self.usar_simulador = usar_simulador
        self.verbose = verbose

        # -------------------------------------------------- ROS 2
        if not rclpy.ok():
            rclpy.init()

        self.nodo = Node('p1_entorno_follow')

        # Publicador de velocidad
        self.pub_vel = self.nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)

        # Suscripciones a sensores
        self._ir = [0] * 8
        self._n_ir = 0
        self.nodo.create_subscription(
            Int32MultiArray, NS_BASE + '/ir', self._cb_ir, 1)

        self._blob_verde = None
        self._n_blobs = 0
        self.nodo.create_subscription(
            BlobArray, NS_SMARTPHONE + '/color_blobs', self._cb_blobs, 1)

        # -------------------------------------------------- Simulador
        self.cli_reset = None
        self._pose = None
        if self.usar_simulador:
            if not HAY_INTERFACES_SIM:
                raise RuntimeError("No se encuentran las interfaces del módulo sim.")
            self.cli_reset = self.nodo.create_client(
                ResetSimulation, NS_SIM + '/reset_simulation')
            if not self.cli_reset.wait_for_service(timeout_sec=5.0):
                raise RuntimeError("El servicio sim/reset_simulation no responde.")
            self.nodo.create_subscription(
                RobotLocation, NS_SIM + '/robot_location', self._cb_pose, 1)

        # -------------------------------------------------- Espacios Gymnasium
        # Vector de observación (dimensión 8):
        # [0] size_norm (0..1)
        # [1] x_norm (0..1)
        # [2] visible (0 o 1)
        # [3] d_size (-1..1)
        # [4] d_x (-1..1)
        # [5] ir_frontal (0..1)
        # [6] v_prev (-1..1)
        # [7] w_prev (-1..1)
        obs_low = np.array([0.0, 0.0, 0.0, -1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Espacio de acción continuo: [v_norm, w_norm] en [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # -------------------------------------------------- Estado interno
        self.pasos = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)
        self.size_prev = 0.0
        self.x_prev = 0.5
        self.tiempo_sin_blob = 0.0

        # Esperar primeras lecturas
        self._esperar_lectura_nueva(inicial=True)

    # -----------------------------------------------------------------
    # Callback ROS 2
    # -----------------------------------------------------------------
    def _cb_ir(self, msg):
        self._ir = list(msg.data)
        self._n_ir += 1

    def _cb_blobs(self, msg):
        blobs_verdes = [b for b in msg.blobs if b.color == 'green']
        if blobs_verdes:
            # Seleccionar el blob verde de mayor tamaño
            self._blob_verde = max(blobs_verdes, key=lambda b: b.size)
        else:
            self._blob_verde = None
        self._n_blobs += 1

    def _cb_pose(self, msg):
        self._pose = (msg.position.x, msg.position.z, msg.rotation.y)

    # -----------------------------------------------------------------
    # Gestión de ciclo de lectura ROS 2
    # -----------------------------------------------------------------
    def _vaciar_cola(self):
        for _ in range(50):
            antes = self._n_blobs
            rclpy.spin_once(self.nodo, timeout_sec=0.0)
            if self._n_blobs == antes:
                return

    def _esperar_lectura_nueva(self, inicial=False):
        objetivo = self._n_blobs + 1
        t0 = time.time()
        while self._n_blobs < objetivo:
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
            if time.time() - t0 > 3.0:
                if inicial:
                    raise RuntimeError("No llegan lecturas de color_blobs.")
                break

    def _publicar_velocidad(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_vel.publish(msg)

    def _detener(self):
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def _reiniciar_escena(self):
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)

    # -----------------------------------------------------------------
    # Observación
    # -----------------------------------------------------------------
    def _observacion(self):
        if self._blob_verde is not None:
            size_norm = min(max(self._blob_verde.size / cfg.BLOB_SIZE_MAX, 0.0), 1.0)
            x_norm = min(max(self._blob_verde.x / cfg.CAM_ANCHO_PX, 0.0), 1.0)
            visible = 1.0
        else:
            size_norm = 0.0
            x_norm = 0.5
            visible = 0.0

        d_size = np.clip(size_norm - self.size_prev, -1.0, 1.0)
        d_x = np.clip(x_norm - self.x_prev, -1.0, 1.0)

        # Proximidad por infrarrojos
        ir_front = float(max(min(max(self._ir[i], 0) / IR_SATURACION, 1.0) for i in IR_FRONTALES))

        obs = np.array([
            size_norm,
            x_norm,
            visible,
            d_size,
            d_x,
            ir_front,
            self.ultima_accion[0],
            self.ultima_accion[1]
        ], dtype=np.float32)

        # Guardar anteriores para el siguiente paso
        self.size_prev = size_norm
        self.x_prev = x_norm

        return obs

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------
    def _recompensa(self, obs, v, w, dw, fallo):
        size_norm, x_norm, visible, _, _, ir_front, _, _ = obs

        # 1. Término por mantener la distancia (Gaussiana sobre el tamaño del blob objetivo)
        # Estimamos que la banda objetivo corresponde a size_norm alrededor de 0.5
        r_dist = cfg.RECOMPENSA['k_separacion'] * math.exp(-((size_norm - 0.5) / 0.2) ** 2)

        # 2. Término por centrado horizontal del blob
        r_desv = -cfg.RECOMPENSA['k_desviacion'] * abs(x_norm - 0.5)

        # 3. Penalización si no se ve el blob
        r_vis = -cfg.RECOMPENSA['k_visibilidad'] if visible == 0.0 else 0.0

        # 4. Penalización por cambios bruscos de giro (suavidad)
        r_suav = -cfg.RECOMPENSA['k_suavidad'] * abs(dw)

        # Recompensa base
        r = r_dist + r_desv + r_vis + r_suav

        # 5. Penalización terminal por fallo (colisión o perder al guía)
        if fallo:
            r -= cfg.RECOMPENSA['penalizacion_fallo']

        return float(r)

    # -----------------------------------------------------------------
    # Interfaz Gymnasium
    # -----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._detener()

        if self.usar_simulador and self.cli_reset is not None:
            self._reiniciar_escena()
            time.sleep(0.5)

        self.pasos = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)
        self.size_prev = 0.0
        self.x_prev = 0.5
        self.tiempo_sin_blob = 0.0

        self._vaciar_cola()
        self._esperar_lectura_nueva()

        return self._observacion(), {}

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        
        # Variación de velocidad angular respecto a la acción anterior
        dw = accion[1] - self.ultima_accion[1]
        self.ultima_accion = accion

        # Reescalar de [-1, 1] a unidades de velocidad real
        v = cfg.V_MIN + (accion[0] + 1.0) * 0.5 * (cfg.V_MAX - cfg.V_MIN)
        w = float(accion[1]) * cfg.W_MAX

        self._publicar_velocidad(v, w)
        time.sleep(cfg.PASO_S)

        self._vaciar_cola()
        self._esperar_lectura_nueva()

        self.pasos += 1
        obs = self._observacion()

        size_norm, x_norm, visible, _, _, ir_front, _, _ = obs

        # Contador para detectar pérdida prolongada del guía (> 3s)
        if visible == 0.0:
            self.tiempo_sin_blob += cfg.PASO_S
        else:
            self.tiempo_sin_blob = 0.0

        # Detección de fallos
        colision = ir_front >= 0.85
        perdido = (self.tiempo_sin_blob >= cfg.TIEMPO_PERDIDO_S)
        fallo = colision or perdido

        recompensa = self._recompensa(obs, v, w, dw, fallo)

        terminated = bool(fallo)
        truncated = bool(self.pasos >= self.pasos_max)

        if terminated or truncated:
            self._detener()

        info = {
            'colision': colision,
            'perdido': perdido,
            'visible': bool(visible),
            'pose_sim': self._pose
        }

        if self.verbose:
            print(f"Paso {self.pasos:3d} | Size: {size_norm:.2f} | X: {x_norm:.2f} | "
                  f"v: {v:+.2f} | w: {w:+.2f} | R: {recompensa:+.2f}")

        return obs, recompensa, terminated, truncated, info

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        try:
            self.nodo.destroy_node()
        except Exception:
            pass