# Práctica 2. Entrega

Jacobo Callejo Iribarren · Francisco Novo Rego

| Contenido | Qué es |
|---|---|
| `RIA_P2_Memoria.pdf` | Memoria de la práctica. |
| `RIA_P2_Declaracion_IA.pdf` | Declaración del uso de IA. |
| `codigo_fuente/` | Código comentado, configuraciones de NEAT y calibración. Su `README.md` explica cómo reproducir las evoluciones y ejecutar el controlador final de cada configuración. |
| `evidencias/p2_runs/` | Una carpeta por evolución de la memoria, con su registro por generación, sus puntos de control, el mejor genoma de cada generación y el genoma final. Además, la calidad del controlador a mano (`referencia_*.json`). |
| `evidencias/p2_eval/` | Registro paso a paso de cada episodio evaluado con el protocolo del anexo B. |
| `evidencias/figuras/` | Figuras del anexo de la memoria. |
| `videos/` | Un vídeo por configuración con su controlador final evaluándose con el protocolo. |

Controladores finales:

| Configuración | Genoma | Carpeta |
|---|---|---|
| C1 | 160 | `evidencias/p2_runs/C1_v5_s2` |
| C2 | 411 | `evidencias/p2_runs/C2_v5_p235_s3` |
| C3 | 411 | `evidencias/p2_runs/C3_v5_p235_s1` |
| C4 | 446 | `evidencias/p2_runs/C4_v6_s1` |

Para ejecutar el código con estos resultados, se copian juntos a la carpeta de trabajo del contenedor:

```bash
cp codigo_fuente/* /docker-ros/ws/
cp -r evidencias/p2_runs evidencias/p2_eval /docker-ros/ws/
```
