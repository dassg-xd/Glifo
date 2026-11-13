# Contrato del FeatureExtractor (2D y 3D)

Convenciones que deben respetar el **HMMEngine** (Integrante 1) y la **GUI** (Integrante 3). Si algo cambia aquí, se avisa a los otros dos antes de tocar el código.

## 1. Valores congelados

| Parámetro | Valor | Por qué |
|---|---|---|
| **K** (sectores) | **8** | K = 16 no mejoró con los datos actuales y exige más datos por modelo |
| **Paso de remuestreo** | **0.03** del lado del lienzo | Meseta de mejor rendimiento (0.02–0.04); da ~25 símbolos por letra |
| **Suavizado** | **1 pasada** | Quita el temblor; 0, 2 y 3 pasadas no se distinguen |
| **Varios trazos** | Orden canónico y un símbolo por salto (`extract_trazos`) | Da igual en qué orden se dibujen los trazos (con el orden invertido el acierto pasó de 31 % a 85 %); el salto del lápiz conserva la posición relativa sin pesar como un trazo |

Cómo se eligieron: se compararon K ∈ {8, 16} y pasos de 0.02 a 0.05 con un clasificador de referencia sencillo (1 vecino más cercano con distancia de edición, **no es el HMM**). K = 8 con paso 0.03–0.04 dio ≈ 95 % y K = 16 ≈ 94 %; las diferencias están dentro del ruido (±1.7 puntos). Son cifras **optimistas**: el reparto fue aleatorio y las muestras de una misma persona se parecen. Sirven para comparar configuraciones, no como precisión final.

**Cambiar K, el paso o el suavizado obliga a regenerar las secuencias y reentrenar los 27 modelos.**

## 2. Símbolos (lo que consume el HMM)

- Alfabeto de observaciones **V = {0, …, 7}**, así que `n_simbolos = 8`.
- El símbolo `s` es el sector angular centrado en `s · 45°`:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| → | ↗ | ↑ | ↖ | ← | ↙ | ↓ | ↘ |

- "Arriba" es el símbolo 2 aunque en pantalla la Y crezca hacia abajo: el extractor invierte el eje internamente.

## 3. Qué debe entregar la GUI

- Lista de puntos `(x, y)` **normalizados**: `x / L`, `y / L`, con `L` = lado del lienzo (lienzo **cuadrado**; si no lo es, dividir ambos ejes por el mismo `L`). El eje Y es el de pantalla.
- **Escala del dataset:** Glifo Capture guarda las coordenadas en una escala de **0 a 1000 relativa al lienzo de captura** (500 es siempre la mitad, en cualquier pantalla; **no son píxeles**). Dividir entre `canvas = 1000` deja los puntos en [0, 1], exactamente la misma escala relativa que debe producir la GUI.
- Lo que importa es la escala **relativa**: una letra típica ocupa ≈ 0.28 del lienzo (rango 0.17–0.47). Si las letras se dibujan mucho más chicas o más grandes que en el dataset, las secuencias saldrán más cortas o más largas que las de entrenamiento.
- **Caso crítico: «i» y «l».** La «i» mide en promedio ≈ 0.16 del lienzo y la «l» ≈ 0.38; su palito da unos 4 símbolos (el punto no aporta) frente a unos 12 de la «l», así que se distinguen sobre todo por el **largo**, que depende de la escala. La GUI debe respetar las mismas proporciones que Glifo Capture (líneas guía: ascendente 0.18, altura x 0.42, línea base 0.70, descendente 0.90).
- Varios trazos: `fx.extract_trazos([trazo1, trazo2, ...])`. Ordena los trazos (principales de izquierda a derecha y al final los pequeños: punto, virgulilla, cruz) y agrega **un** símbolo con la dirección de cada salto del lápiz. El entrenamiento usa la misma función.

## 4. API

```python
from src.FeatureExtractor import FeatureExtractor, ALPHABET

fx = FeatureExtractor()                 # K=8, step=0.03, smooth_passes=1

fx.extract(puntos)                      # -> list[int]  símbolos O_1..O_T
fx.extract_2d(puntos)                   # alias de extract(puntos, "2d")
fx.preprocess(puntos)                   # puntos suavizados y equiespaciados (depurar/dibujar)
fx.symbols_to_points(restantes, start)  # -> list[(x, y)]  proyección punteada
fx.extract_trazos(trazos)               # varios trazos -> símbolos (orden canónico + 1 símbolo por salto)
FeatureExtractor.ordenar_trazos(trazos) # solo el orden canónico
FeatureExtractor.join_strokes(trazos)   # varios trazos -> una polilínea (ya no se usa para extraer)
```

- `mode` es opcional y vale `"2d"` o `"3d"` (ver sección 8); cualquier otro valor lanza `ValueError`.
- `k` solo admite 8 o 16. Puntos con NaN/inf, con otra dimensión o no numéricos lanzan `ValueError`, igual que símbolos fuera de rango en `symbols_to_points`.
- `ALPHABET`: las 27 letras, con `ñ`.
- Usar siempre los valores por defecto del extractor, **sin copiarlos a mano**, para que entrenamiento y GUI coincidan.

## 5. Uso en tiempo real (GUI)

1. En cada evento del ratón, añadir el punto (ya normalizado) a la lista.
2. Llamar a `fx.extract(puntos)` con la lista **completa**: es barato y devuelve `O_1:t`.
3. El HMMEngine entrega los símbolos restantes `{O_t+1..O_T}` de la letra ganadora y la GUI llama a `fx.symbols_to_points(restantes, start=ultimo_punto)`.
4. Dibujar esos puntos con línea punteada.

Notas:
- El último tramo menor a un paso no produce símbolo, así que la secuencia crece "a saltos".
- Al llegar un punto nuevo, el último símbolo puede cambiar (el suavizado mira a los vecinos). Recalcular desde cero con la lista completa lo resuelve.
- Con menos de ~4 símbolos la clasificación es poco fiable; conviene no mostrar ranking hasta tener algunos.

## 6. Datos

**Entrada** (la genera Glifo Capture): `data/glifo-dataset-AAAA-MM-DD.json`

```json
{ "a": [ [ [ [481.7, 511.8, 0], [472.9, 505.9, 12] ], [ ...otro trazo... ] ], ... ], "b": [ ... ] }
```
`letra → muestras → trazos → [x, y, t_ms]`, con `x`, `y` en la escala **0–1000 relativa al lienzo de captura** (no píxeles). El tiempo se descarta.

**Salida** de `python training/build_sequences.py` → `data/secuencias_2d_k8.json`:

```json
{ "modo": "2d", "k": 8, "n_simbolos": 8, "paso": 0.03, "suavizado": 1,
  "canvas": 1000.0, "holdout": 0.2, "semilla": 42,
  "secuencias": { "a": [[0, 0, 1, 2, ...], ...], "b": [...], ... },
  "prueba":     { "a": [[...], ...], ... } }
```

- `secuencias` es el conjunto de **entrenamiento** y `prueba` el de **evaluación** (20 % por letra, reproducible con la semilla). Con `--holdout 0` todo va a entrenamiento (para el modelo final).
- Es un archivo **generado**: regenerarlo cada vez que cambie el dataset, K, el paso o el suavizado.
- El reparto es aleatorio porque el dataset estándar no trae autor; con la descarga "con detalles" de Glifo Capture se podrá separar por persona (segunda entrega).
- El modelo exportado por el HMMEngine debería guardar `k`, `paso` y el número de estados para detectar desajustes con la GUI.

## 7. Comandos útiles

```bash
python training/check_data.py                        # calidad del dataset
python training/build_sequences.py                   # dataset -> secuencias (K=8)
python training/build_sequences.py --holdout 0       # todo a entrenamiento
python -m unittest discover -s tests -v              # pruebas
```

## 8. Modo 3D (esfera)

### Marco de la vista

Los puntos llegan como `(x, y, z)` sobre la esfera unitaria, en el **marco de la vista**: `x` a la derecha, `y` hacia arriba de la pantalla, `z` hacia quien mira. La GUI guarda los trazos pegados a la esfera y, antes de analizar, los multiplica por la rotación actual (`v = R · p`). Con eso:

- **«Arriba» es siempre el borde superior del panel**, gire como gire la esfera.
- **Solo cuenta lo que se ve** (`z > 0`). Un trazo que pasa por detrás queda partido en sus tramos visibles, que se unen con un símbolo de salto, igual que dos trazos.

### De (x, y, z) a símbolos

| Paso | Fórmula |
|---|---|
| Coordenadas esféricas | θ = atan2(x, z) (acimut, crece a la derecha) · φ = arccos(y) (polar, 0 arriba y π/2 en el centro de la vista) |
| Variaciones | Δθ (llevado a [−π, π]) y Δφ entre dos puntos seguidos |
| Rumbo sobre la superficie | este = sin φ̄ · Δθ · norte = −Δφ · rumbo = atan2(norte, este) |
| Símbolo | `quantize_angle(rumbo, K)`: los mismos 8 símbolos del 2D |

El factor sin φ̄ (φ̄ = promedio de los dos φ) hace falta porque, lejos del centro, un mismo Δθ recorre menos distancia: sin él, una diagonal parecería horizontal.

### Remuestreo y tamaño

- El remuestreo es por **longitud de arco sobre círculos máximos**, con un paso **angular** (interpolación esférica entre puntos).
- `ANGULO_LIENZO = 1.6` rad equivale al lado del lienzo 2D sin zoom; el paso por defecto es `0.03 × 1.6 = 0.048` rad. Así una letra de tamaño cómodo da tantos símbolos como en el plano (la «a» promedia 29.9 símbolos en 3D y 30.2 en 2D).
- Con zoom `z` la GUI usa `paso = 0.048 / z`: una letra del mismo tamaño en pantalla produce los mismos símbolos.

### API 3D

```python
fx = FeatureExtractor()                              # paso_angular = 0.048 rad
fx.extract_trazos(trazos_vista, "3d")                # trazos (x, y, z) -> símbolos (solo lo visible)
fx.extract(puntos_vista, "3d")                       # un trazo
fx.preprocess(puntos_vista, "3d")                    # puntos suavizados y equiespaciados sobre la esfera
fx.symbols_to_points(simbolos, inicio, "3d")         # camina por círculos máximos desde `inicio`
FeatureExtractor.ordenar_trazos_3d(trazos)           # orden canónico: θ en lugar de x, φ en lugar de y

a_esfericas(p) / de_esfericas(theta, phi)            # (x, y, z) <-> (θ, φ)
rumbo(a, b)                                          # dirección de a hacia b con (Δθ, Δφ)
mover(p, rumbo, distancia)                           # avanzar por el círculo máximo
visibles(trazo)                                      # tramos con z > 0
pegar_en_esfera(trazos_2d, theta0, phi0, lado, giro) # lienzo 2D -> esfera (mapa exponencial)
```

Los puntos 3D se normalizan a la esfera; el origen, NaN, infinitos u otra dimensión lanzan `ValueError`.

### Dataset y modelos 3D

`python training/build_sequences_3d.py` pega cada muestra 2D sobre la cara visible con `pegar_en_esfera` (centro θ ∈ ±0.35, φ ∈ π/2 ± 0.30; tamaño ±30 %; giro ±8°), 4 veces las de entrenamiento y 2 las de prueba, y escribe `data/secuencias_3d_k8.json`. `python training/entrenar_hmms.py --modo 3d` entrena `data/modelos_letras_3d.json`. **Cambiar el paso, `ANGULO_LIENZO` o los rangos obliga a regenerar y reentrenar.**

