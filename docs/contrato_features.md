# Contrato del FeatureExtractor (2D y 3D)

Convenciones que deben respetar el **HMMEngine** (Integrante 1) y la **GUI** (Integrante 3). Si algo cambia aquí, se avisa a los otros dos antes de tocar el código.

## 1. Valores congelados

| Parámetro | Valor | Por qué |
|---|---|---|
| **K** (sectores) | **16** | Con el dataset de un diseño por letra, 16 direcciones separan mejor curvas y diagonales (ver la tabla de abajo) |
| **Zonas verticales** | **3**, con el borde a **4 pasos** del inicio de la letra | Le dan al HMM la altura del trazo: separan «h»/«n», «b»/«p», «d»/«q» |
| **Paso de remuestreo** | **0.03** del lado del lienzo | Meseta de mejor rendimiento (0.02–0.04); da ~25 símbolos por letra |
| **Suavizado** | **1 pasada** | Quita el temblor; 0, 2 y 3 pasadas no se distinguen |
| **Varios trazos** | Orden canónico y un símbolo por salto (`extract_trazos`) | Da igual en qué orden se dibujen los trazos (con el orden invertido el acierto pasó de 31 % a 85 %); el salto del lápiz conserva la posición relativa sin pesar como un trazo |

El paso y el suavizado se eligieron primero con un clasificador de referencia sencillo (1 vecino más cercano con distancia de edición, **no es el HMM**), comparando pasos de 0.02 a 0.05. K y las zonas se eligieron después con el propio HMM.

Con el HMM, sobre una validación apartada del entrenamiento (495 muestras; «app» = con los términos de fin y de largo):

| Configuración | Solo HMM | App |
|---|---|---|
| K = 8 | 90.9 % | 93.9 % |
| K = 16 | 91.7 % | 95.8 % |
| K = 8 × 3 zonas | 95.4 % | 96.4 % |
| **K = 16 × 3 zonas** | **96.8 %** | **98.0 %** |
| K = 16 × 4 zonas | 96.8 % | 97.4 % |
| K = 8, hasta 12 estados | 90.3 % | 94.3 % |

**Cambiar K, las zonas, el paso o el suavizado obliga a regenerar las secuencias y reentrenar los 27 modelos.**

## 2. Símbolos (lo que consume el HMM)

- Alfabeto de observaciones **V = {0, …, 47}**: `n_simbolos = K · 3 = 48`.
- `símbolo = dirección + K · zona`. `fx.direccion(s)` y `fx.zona(s)` los separan.
- **Dirección** (0 a 15): el sector angular centrado en `d · 22.5°`. 0 = →, 4 = ↑, 8 = ←, 12 = ↓. «Arriba» es 4 aunque en pantalla la Y crezca hacia abajo: el extractor invierte el eje internamente.
- **Zona** (0 a 2): altura del punto medio del segmento respecto del **primer punto de la letra** (el inicio del primer trazo en el orden canónico). 0 = más de 4 pasos por encima, 1 = a esa altura (±4 pasos), 2 = más de 4 pasos por debajo. La zona de un salto del lápiz es la de su destino. Como se mide desde el inicio de la letra, no depende de en qué parte del lienzo se dibuje.
- `fx.extract(puntos)` sigue dando el código de cadena puro de **un** trazo (solo direcciones, 0 a K−1). Los símbolos con zona salen de `fx.extract_trazos(trazos)`, que es lo que usan el entrenamiento y la GUI.

## 3. Qué debe entregar la GUI

- Lista de puntos `(x, y)` **normalizados**: `x / L`, `y / L`, con `L` = lado del lienzo (lienzo **cuadrado**; si no lo es, dividir ambos ejes por el mismo `L`). El eje Y es el de pantalla.
- **Escala del dataset:** Glifo Capture guarda las coordenadas en una escala de **0 a 1000 relativa al lienzo de captura** (500 es siempre la mitad, en cualquier pantalla; **no son píxeles**). Dividir entre `canvas = 1000` deja los puntos en [0, 1], exactamente la misma escala relativa que debe producir la GUI.
- Lo que importa es la escala **relativa**: una letra típica ocupa ≈ 0.28 del lienzo (rango 0.17–0.47). Si las letras se dibujan mucho más chicas o más grandes que en el dataset, las secuencias saldrán más cortas o más largas que las de entrenamiento.
- **Caso crítico: «i» y «l».** La «i» mide en promedio ≈ 0.16 del lienzo y la «l» ≈ 0.38; su palito da unos 4 símbolos (el punto no aporta) frente a unos 12 de la «l», así que se distinguen sobre todo por el **largo**, que depende de la escala. La GUI debe respetar las mismas proporciones que Glifo Capture (líneas guía: ascendente 0.18, altura x 0.42, línea base 0.70, descendente 0.90).
- Varios trazos: `fx.extract_trazos([trazo1, trazo2, ...])`. Ordena los trazos (principales de izquierda a derecha y al final los pequeños: punto, virgulilla, cruz) y agrega **un** símbolo con la dirección de cada salto del lápiz. El entrenamiento usa la misma función.

## 4. API

```python
from src.FeatureExtractor import FeatureExtractor, ALPHABET

fx = FeatureExtractor()                 # K=16, step=0.03, smooth_passes=1

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

**Salida** de `python training/build_sequences.py` → `data/secuencias_2d_k16.json`:

```json
{ "modo": "2d", "k": 16, "n_simbolos": 48, "paso": 0.03, "suavizado": 1,
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
python training/build_sequences.py                   # dataset -> secuencias (K=16, 3 zonas)
python training/build_sequences.py --holdout 0       # todo a entrenamiento
python -m unittest discover -s tests -v              # pruebas
```

## 8. Modo 3D (esfera)

### Marco de la esfera

Los puntos llegan como `(x, y, z)` sobre la esfera unitaria, en el **marco de la propia esfera**, tal como la GUI los guarda pegados a la superficie. La rotación de la vista no interviene en el análisis:

- **«Arriba» es el polo norte de la esfera** (su eje `+y`): subir es ir hacia el polo por el meridiano. No depende del panel ni de cómo esté girada la esfera.
- **Cuenta toda la esfera**, se vea o no. Un trazo que pasa por la cara de atrás sigue siendo un solo trazo.

### De (x, y, z) a símbolos

| Paso | Fórmula |
|---|---|
| Coordenadas esféricas | θ = atan2(x, z) (longitud, crece hacia el este) · φ = arccos(y) (polar, 0 en el polo norte y π/2 en el ecuador) |
| Variaciones | Δθ (llevado a [−π, π]) y Δφ entre dos puntos seguidos |
| Rumbo sobre la superficie | este = sin φ̄ · Δθ · norte = −Δφ · rumbo = atan2(norte, este) |
| Símbolo | `quantize_angle(rumbo, K) + K · zona`: el mismo alfabeto del 2D; la zona se mide con φ (abajo = alejarse del polo norte) y su borde es 4 pasos angulares |

El factor sin φ̄ (φ̄ = promedio de los dos φ) hace falta porque, lejos del ecuador, un mismo Δθ recorre menos distancia: sin él, una diagonal parecería horizontal.

### Remuestreo y tamaño

- El remuestreo es por **longitud de arco sobre círculos máximos**, con un paso **angular** (interpolación esférica entre puntos).
- `ANGULO_LIENZO = 1.6` rad equivale al lado del lienzo 2D sin zoom; el paso por defecto es `0.03 × 1.6 = 0.048` rad. Así una letra de tamaño cómodo da tantos símbolos como en el plano (la «a» promedia 29.9 símbolos en 3D y 30.2 en 2D).
- Con zoom `z` la GUI usa `paso = 0.048 / z`: una letra del mismo tamaño en pantalla produce los mismos símbolos.

### API 3D

```python
fx = FeatureExtractor()                              # paso_angular = 0.048 rad
fx.extract_trazos(trazos, "3d")                      # trazos (x, y, z) -> símbolos (toda la esfera)
fx.extract(puntos, "3d")                             # un trazo
fx.preprocess(puntos, "3d")                          # puntos suavizados y equiespaciados sobre la esfera
fx.symbols_to_points(simbolos, inicio, "3d")         # camina por círculos máximos desde `inicio`
FeatureExtractor.ordenar_trazos_3d(trazos)           # orden canónico: θ (desde el centro de la letra) en lugar de x, φ en lugar de y

a_esfericas(p) / de_esfericas(theta, phi)            # (x, y, z) <-> (θ, φ)
rumbo(a, b)                                          # dirección de a hacia b con (Δθ, Δφ)
mover(p, rumbo, distancia)                           # avanzar por el círculo máximo
centro_esferico(trazos)                              # centro de la letra sobre la esfera
pegar_en_esfera(trazos_2d, theta0, phi0, lado, giro) # lienzo 2D -> esfera (mapa exponencial)
```

Los puntos 3D se normalizan a la esfera; el origen, NaN, infinitos u otra dimensión lanzan `ValueError`.

### Dataset y modelos 3D

`python training/build_sequences_3d.py` pega cada muestra 2D sobre la esfera con `pegar_en_esfera` (centro en cualquier longitud, θ ∈ ±π, y φ ∈ π/2 ± 0.60; tamaño ±30 %; giro ±8°), 4 veces las de entrenamiento y 2 las de prueba, y escribe `data/secuencias_3d_k16.json`. `python training/entrenar_hmms.py --modo 3d` entrena `data/modelos_letras_3d.json`. **Cambiar el paso, `ANGULO_LIENZO` o los rangos obliga a regenerar y reentrenar.**

