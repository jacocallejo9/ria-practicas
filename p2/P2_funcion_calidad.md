# P2 - Función de calidad

## Fórmula

La calidad se acumula de forma **mixta**: una parte se suma en cada paso y al final se añade una bonificación si el episodio se completa.

```
F = 20 · n/N  +  70 · (1/N) · Σ q_t  +  10 · [completado]
```

- N = 100 es el número máximo de pasos del episodio (~60 s). La suma va desde el primer paso hasta el último que se llega a vivir.
- n es el número de pasos vividos sin fallar (supervivencia).
- q_t es la calidad del paso t, entre 0 y 1. Se explica más abajo.
- [completado] vale 1 si el episodio llega a los N pasos sin fallar y 0 si no.
- F va de 0 a 100.

Se divide entre N y no entre los pasos vividos. Así, si el robot falla antes, los pasos que le faltan cuentan como 0, y esa es la penalización por fallar: no ponemos ninguna penalización aparte.

El término de supervivencia (20 puntos) se añadió tras la primera evolución de C1. Sin él, con el producto de la formulación A, un individuo que aguanta pero va mal orientado sumaba casi 0, igual que uno que choca en el primer segundo: en la generación 0 casi todos tenían F < 2 y la selección era prácticamente aleatoria. Con él, perder al guía a los 8 s (≈ 12 puntos) puntúa claramente más que perderlo a los 2 s (≈ 3). No se puede explotar quedándose quieto ni girando: en C1-C3 el guía no espera, así que el que no le sigue acaba en "guía perdido" o "sin pelota".

Los 10 puntos de bonificación hacen que completar el episodio valga más que hacerlo casi perfecto y fallar al final (88,4).

## Calidad de cada paso

En cada paso se miran tres cosas, cada una con un valor entre 0 y 1. Dependen de datos del simulador que la red no recibe: la separación real, la desviación real y la posición respecto al carril.

**Separación, C_sep(d).** d es la separación real entre el centro del seguidor y la pelota (posiciones del simulador). Vale 1 dentro de la banda objetivo (0,20-0,45 m). Por debajo baja en línea recta hasta 0 en 0,12 m (colisión), y por encima hasta 0 en 0,90 m (guía perdido). Así un individuo que se queda cerca de la banda puntúa más que uno que está lejos.

**Orientación, C_ori(θ).** θ es la desviación real, el ángulo entre el rumbo del seguidor y la dirección a la pelota (simulador). Vale 1 si |θ| ≤ 15°, que es el umbral del protocolo, y baja hasta 0 en |θ| = 23,3°. Ese ángulo es la mitad del campo de visión de la cámara: a partir de ahí la pelota se sale de la imagen.

**Carril, C_car(e).** e es el error lateral, la distancia del seguidor a la línea central de su carril (la línea grabada en la P1). Vale 1 si |e| ≤ 0,045 m, que es cuando el robot (0,16 m) cabe entero en el carril (0,25 m). Baja hasta 0 en |e| = 0,125 m, cuando el centro del robot pisa la raya. Es la tolerancia que pide el enunciado para el carril.

Probaremos dos formas de juntar los tres términos:

- A (producto): q_t = C_sep · C_ori · C_car. El paso sólo puntúa bien si se cumplen las tres cosas a la vez.
- B (suma): q_t = 0,5 · C_sep + 0,3 · C_ori + 0,2 · C_car. Es más permisiva. Damos más peso a la separación porque es el requisito principal.

## Duración de la evaluación

Cada individuo se evalúa en un episodio de como máximo 100 pasos, unos 60 s (en la P1 un paso duraba ~0,6 s). Elegimos esta duración para que todas las configuraciones tengan al menos una curva. En C1, que es la más lenta, el guía no empieza a girar hasta los 40 s más o menos.

Todos los episodios empiezan con un reinicio de la escena, como en el protocolo (anexo B.2). En la primera evolución el 70 % empezaba colocando al seguidor detrás del guía donde estuviera ("mixto"): cada individuo se evaluaba en un tramo distinto (recta o curva) y la calidad dependía más de la suerte que del individuo.

## Cuándo se corta el episodio

El episodio termina antes de tiempo si pasa algo de esto:

- Colisión: separación menor que 0,12 m.
- Guía perdido: separación mayor que 0,90 m.
- No se ve la pelota durante más de 3 s seguidos.
- El seguidor se sale de la calzada. No queremos ese comportamiento, y además cortarlo ahorra tiempo de evaluación.

Si llega a los 100 pasos, el episodio termina como completado.

Si el simulador se congela o se cae el puente, no es culpa del individuo: no le ponemos un 0, sino que repetimos su evaluación.
