# Memoria P2 en LaTeX

- `memoria.tex`: la memoria. Compilar con **pdfLaTeX** (en Overleaf: subir esta carpeta entera, Menu → Compiler: pdfLaTeX).
  Usa Carlito, que tiene las mismas métricas que Calibri 11 pt (Calibri no está en TeX Live).
- `figuras/`: figuras en PDF (y las topologías en `.dot`).
- `figuras_memoria.py`: regenera las figuras a partir de `p2_runs/` y `p2_eval/`. Desde `p2/p2_fran`, dentro del contenedor:

```bash
PYTHONNOUSERSITE=1 python3 ../memoria_latex/figuras_memoria.py --salida ../memoria_latex/figuras
for f in ../memoria_latex/figuras/red_*.dot; do dot -Tpdf "$f" -o "${f%.dot}.pdf"; done
```

`PYTHONNOUSERSITE=1` hace falta porque el matplotlib de apt no funciona con el numpy 2 instalado con pip --user.
Los datos de C3_v5_p235_s1/s2 y sus CSV del protocolo están todavía sólo en `docker-ros/ws`: hay que copiarlos a `p2_fran` para que el script los encuentre.

Lo marcado en rojo como `\pendiente{...}` son datos que faltan (C4 final, protocolo a mano en C1, SAC en la misma sesión, transferencia al real, vídeos).
