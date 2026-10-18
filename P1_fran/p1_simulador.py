#!/usr/bin/env python3
"""
Práctica 1 — Canal privilegiado con RoboboSim (verdad del terreno y escena).

Todo lo que es del ROBOT (cámara, IR, cmd_vel) pasa por ROS 2 a través del
nodo puente. Lo que sólo existe en el simulador (posiciones reales,
reinicio de la escena y colocación del seguidor) se pide aquí, con una
conexión DIRECTA al simulador (librería robobosim, puerto 50505), en vez
de con el módulo sim del puente. Motivos, medidos con p1_diagnostico.py y
el registro del simulador (Player.log):

  - Pedir sim/reset_simulation a través del puente hace que RoboboSim
    cierre la conexión del propio puente ("ObjectDisposedException ...
    Closed websocket connection"): desde ese momento el módulo sim publica
    posiciones congeladas y ya no reinicia nada. Con una conexión propia
    se detecta (connectionState) y se reconecta.
  - En Follow City el guía y su pelota son objetos de la escena y sus
    posiciones llegan unas 300 veces por segundo; el puente las reenvía
    todas por ROS 2 y se satura ("Too much concurrent calls"). Aquí sólo se
    leen cuando hacen falta.

Nada de esto entra en la observación de la política.

Convenciones: el simulador trabaja en mm y grados con el suelo en el
plano (x, z); aquí se devuelven metros.
"""

import contextlib
import io
import time

import numpy as np

try:
    from robobosim.RoboboSim import RoboboSim
    from robobosim.utils.ConnectionState import ConnectionState
    HAY_ROBOBOSIM = True
except ImportError:
    RoboboSim = ConnectionState = None
    HAY_ROBOBOSIM = False


class ClienteSim:

    def __init__(self, ip, robot_id, objeto_guia, objeto_pelota=None):
        if not HAY_ROBOBOSIM:
            raise RuntimeError('Falta la librería robobosim:  pip install --user robobosim')
        self.ip = ip
        self.robot_id = int(robot_id)
        self.objeto_guia = objeto_guia
        self.objeto_pelota = objeto_pelota
        self.sim = None
        self.conectar()

    # -----------------------------------------------------------------
    # Conexión
    # -----------------------------------------------------------------

    def conectar(self):
        """Abre la conexión y espera a recibir posiciones (máx. 5 s)."""
        self.cerrar()
        self.sim = RoboboSim(self.ip)
        with contextlib.redirect_stdout(io.StringIO()):   # la librería imprime "wait"...
            self.sim.connect()
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self.sim.getRobotLocation(self.robot_id) and \
                    self.sim.getObjectLocation(self.objeto_guia):
                return
            time.sleep(0.05)
        raise RuntimeError(
            'RoboboSim no envía la posición del robot {} o del objeto "{}". ¿Está '
            'abierto el mundo Follow City?'.format(self.robot_id, self.objeto_guia))

    def conectado(self):
        return self.sim is not None and self.sim.rem.connectionState == ConnectionState.CONNECTED

    def asegurar(self):
        """Reconecta si el simulador ha cerrado la conexión."""
        if not self.conectado():
            print('[sim] conexión con RoboboSim perdida: reconectando...')
            self.conectar()

    def cerrar(self):
        if self.sim is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.sim.disconnect()
            except Exception:
                pass
        self.sim = None

    # -----------------------------------------------------------------
    # Lectura (verdad del terreno)
    # -----------------------------------------------------------------

    def seguidor(self):
        """(x m, z m, rotación y en grados, altura y en mm) del seguidor."""
        self.asegurar()
        loc = self.sim.getRobotLocation(self.robot_id)
        p, r = loc['position'], loc['rotation']
        return p['x'] / 1000.0, p['z'] / 1000.0, r['y'], p['y']

    def objeto(self, nombre):
        """Posición (x, z) en metros de un objeto de la escena."""
        self.asegurar()
        loc = self.sim.getObjectLocation(nombre)
        if loc is None:
            raise RuntimeError('El simulador no conoce el objeto "{}".'.format(nombre))
        return np.array([loc['position']['x'], loc['position']['z']]) / 1000.0

    def guia(self):
        return self.objeto(self.objeto_guia)

    def pelota(self):
        return self.objeto(self.objeto_pelota) if self.objeto_pelota else None

    # -----------------------------------------------------------------
    # Escena
    # -----------------------------------------------------------------

    def reiniciar(self):
        """Reinicia la escena: los robots vuelven a su posición inicial.

        En Follow City el guía queda parado hasta que el seguidor se acerca
        a menos de ~0,35-0,40 m de la pelota (medido con p1_diagnostico).
        """
        self.asegurar()
        self.sim.resetSimulation()
        time.sleep(0.3)
        if not self.conectado():
            self.conectar()

    def colocar_seguidor(self, x, z, yaw_deg, altura_mm=None):
        """Coloca al seguidor en (x, z) metros con rotación y en grados."""
        self.asegurar()
        if altura_mm is None:
            altura_mm = self.seguidor()[3]
        self.sim.setRobotLocation(self.robot_id,
                                  position={'x': float(x * 1000), 'y': float(altura_mm),
                                            'z': float(z * 1000)},
                                  rotation={'x': 0.0, 'y': float(yaw_deg), 'z': 0.0})
