#!/usr/bin/env python3
"""
p1_calibrar.py — Herramienta de caracterización del detector de blobs.

Avanza el robot seguidor hacia la pelota verde del guía a pulsos cortos
para medir cómo varían el área ('size') y el centro 'X' del blob en píxeles
con respecto a la distancia física real en metros (vía sim/robot_location).
"""

import time
import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from robobo_ros2_interfaces.msg import BlobArray

try:
    from robobo_ros2_interfaces.msg import RobotLocation
    HAY_SIM = True
except ImportError:
    HAY_SIM = False

NS_BASE = '/robobo/robot_0/base'
NS_SMARTPHONE = '/robobo/robot_0/smartphone'
NS_SIM = '/robobo/robot_0/sim'


def ejecutar_calibracion():
    if not rclpy.ok():
        rclpy.init()

    nodo = Node('p1_calibracion_blobs')

    datos = {
        'blob_verde': None,
        'pose': None,
        'n_blobs': 0
    }

    def cb_blobs(msg):
        verdes = [b for b in msg.blobs if b.color == 'green']
        if verdes:
            datos['blob_verde'] = max(verdes, key=lambda b: b.size)
        else:
            datos['blob_verde'] = None
        datos['n_blobs'] += 1

    def cb_pose(msg):
        # x, z en milímetros desde el módulo sim
        datos['pose'] = (msg.position.x / 1000.0, msg.position.z / 1000.0)

    nodo.create_subscription(BlobArray, NS_SMARTPHONE + '/color_blobs', cb_blobs, 1)
    if HAY_SIM:
        nodo.create_subscription(RobotLocation, NS_SIM + '/robot_location', cb_pose, 1)

    pub_vel = nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)

    def esperar_lectura():
        objetivo = datos['n_blobs'] + 1
        t0 = time.time()
        while datos['n_blobs'] < objetivo:
            rclpy.spin_once(nodo, timeout_sec=0.05)
            if time.time() - t0 > 3.0:
                raise RuntimeError("No se reciben mensajes de color_blobs.")

    def mover(v, t):
        msg = Twist()
        msg.linear.x = float(v)
        t0 = time.time()
        while time.time() - t0 < t:
            pub_vel.publish(msg)
            time.sleep(0.05)
        msg.linear.x = 0.0
        for _ in range(3):
            pub_vel.publish(msg)
            time.sleep(0.05)

    print("\n--- INICIANDO CARACTERIZACIÓN DEL BLOB VERDE ---")
    print(f"{'Paso':>4s} | {'Dist. Real (m)':>14s} | {'Blob Size (px)':>14s} | {'Blob X (px)':>12s}")
    print("-" * 55)

    try:
        esperar_lectura()
        pose_inicial = datos['pose']

        for paso in range(1, 101):
            mover(v=0.04, t=0.2)  # Avance a pulsos de 0.2 s
            esperar_lectura()

            blob = datos['blob_verde']
            pose_actual = datos['pose']

            if pose_inicial and pose_actual:
                dist_recorrida = math.hypot(
                    pose_actual[0] - pose_inicial[0],
                    pose_actual[1] - pose_inicial[1]
                )
                dist_str = f"{dist_recorrida:14.3f}"
            else:
                dist_str = "            --"

            if blob is not None:
                size_str = f"{blob.size:14.1f}"
                x_str = f"{blob.x:12.1f}"
            else:
                size_norm = "NO VISIBLE"
                size_str = f"{'NO VISIBLE':>14s}"
                x_str = f"{'--':>12s}"

            print(f"{paso:4d} | {dist_str} | {size_str} | {x_str}")

            # Detener la calibración si la pelota está excesivamente cerca (prevención de colisión)
            if blob and blob.size > 800:
                print("\n[ALERTA] Tamaño máximo alcanzado (cerca del límite de colisión).")
                break

    except KeyboardInterrupt:
        print("\nCalibración interrumpida por el usuario.")
    finally:
        mover(v=-0.05, t=1.0)  # Retroceder un poco al terminar
        nodo.destroy_node()


if __name__ == '__main__':
    ejecutar_calibracion()