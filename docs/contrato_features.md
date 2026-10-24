# Contrato del FeatureExtractor (Integrante 2)

Este documento fija las convenciones que deben respetar el **HMMEngine** (Integrante 1) y la **GUI** (Integrante 3). Si algo cambia aquí, hay que avisar a los otros dos.

## 1. Símbolos (lo que consume el HMM)

- El alfabeto de observaciones es **V = {0, …, K−1}**, con **K = 8** por defecto (también se prueba K = 16). Por tanto `n_simbolos = K`.
- El símbolo `s` es el sector angular centrado en `s · 360°/K`.
- Con K = 8: `0=→  1=↗  2=↑  3=↖  4=←  5=↙  6=↓  7=↘`

## 2. Qué debe entregar la GUI

### 2D
- Lista de puntos `(x, y)` **normalizados**: `x / L`, `y / L`, donde `L` es el lado del lienzo (usar lienzo cuadrado, o dividir ambos ejes por `min(ancho, alto)`).
- El eje Y es el de pantalla (crece hacia abajo). El extractor ya lo invierte internamente.

### 3D
- Lista de puntos `(x, y, z)` sobre la **esfera unitaria**, en el **marco de la cámara** (después de aplicar la rotación actual de la vista): `x` → derecha, `y` → arriba, `z` → hacia el observador.
- Así los datos no dependen de cómo el usuario haya girado la esfera, y el centro de lo visible no cae sobre un polo.
- El grabador (`training/recorder.py`) usa exactamente este marco (proyección ortográfica inversa), por lo que sus datos sirven para entrenar el modo 3D.

### Varios trazos (levantar el lápiz)
`FeatureExtractor.join_strokes([trazo1, trazo2, ...])` los une en una sola polilínea antes de extraer.

## 3. API

```python
from src.FeatureExtractor import FeatureExtractor, ALPHABET

fx = FeatureExtractor(k=8, step_2d=0.04, step_3d=0.08, smooth_passes=1)

fx.extract_2d(puntos_2d) -> list[int]          # símbolos O_1..O_T
fx.extract_3d(puntos_3d) -> list[int]
fx.extract(puntos, mode="2d" | "3d")           # atajo

fx.symbols_to_points(simbolos, start, mode="2d" | "3d") -> list[punto]   # proyección punteada
fx.preprocess(puntos, mode)                    # puntos suavizados y equiespaciados (para depurar/dibujar)
```

- `ALPHABET`: las 27 letras (con `ñ`).
- El **paso** de remuestreo es una fracción del lienzo en 2D y un ángulo en radianes en 3D. **Debe ser el mismo en entrenamiento y en la GUI** (usar los valores por defecto del extractor, no copiarlos a mano).

## 4. Uso en tiempo real (GUI)

1. En cada evento del ratón, añade el punto a la lista (ya normalizado).
2. Llama a `fx.extract(puntos, mode)` con la lista completa: es barato. Devuelve `O_1:t`.
3. Para la proyección, el HMMEngine entrega los símbolos restantes `{O_t+1..O_T}` y la GUI llama a  
   `fx.symbols_to_points(restantes, start=ultimo_punto_del_cursor, mode=mode)`.
4. Dibuja esos puntos con línea punteada.

Notas:
- El último tramo menor a un paso no produce símbolo, así que la secuencia crece "a saltos"; es lo esperado.
- Al llegar puntos nuevos, el último símbolo puede cambiar (el suavizado mira vecinos). Recalcular desde cero con la lista completa lo resuelve.
- La proyección usa el mismo paso que el entrenamiento, así que supone una letra de tamaño parecido al de los datos grabados.

## 5. Datos

- Muestras crudas: `data/<modo>/<letra>/<autor>_<fecha>.json` (puntos crudos, no símbolos). La `ñ` va en la carpeta `enie`.
- Secuencias para entrenar: `python training/build_sequences.py --modo 2d --k 8` genera `data/secuencias_2d_k8.json`:

```json
{ "modo": "2d", "k": 8, "paso": 0.04, "n_simbolos": 8,
  "secuencias": { "a": [[0, 0, 1, 2, ...], ...], "b": [...], ... } }
```

## 6. Comandos útiles

```bash
python training/recorder.py --autor tunombre            # grabar trazos 2D
python training/recorder.py --autor tunombre --modo 3d  # grabar trazos en esfera
python training/build_sequences.py --modo 2d --k 8      # trazos -> secuencias
python -m unittest discover -s tests -v                 # pruebas
```
