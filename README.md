# Glifo

**Reconocedor de letras manuscritas en tiempo real (2D y 3D) con Cadenas Ocultas de Markov.**

Proyecto Final · Procesos Estocásticos (603504) · Universidad de los Llanos · Ingeniería de Sistemas

> **Estado:** funcionan los dos modos: **2D** (lienzo plano) y **3D** (esfera que se gira y se acerca).

---

## ¿Qué hace?

Dibujas una letra con el ratón y la aplicación:

1. **Adivina** qué letra es, mientras la dibujas (27 letras minúsculas del alfabeto español).
2. **Muestra** un ranking con la probabilidad de cada letra.
3. **Autocompleta** el trazo con una línea punteada que predice cómo seguiría la letra.

Funciona en dos modos: sobre un **plano 2D** y sobre una **esfera 3D**.

---

## La idea: ¿por qué un HMM?

Una letra se dibuja como una secuencia de movimientos: "abajo, abajo, derecha, arriba...". Cada letra tiene su propio patrón.

- **Lo que vemos (observaciones):** la dirección del trazo en cada instante.
- **Lo que no vemos (estados ocultos):** en qué parte de la letra va el usuario (inicio, curva, cierre...).

Un **HMM** modela justamente eso. Entrenamos **uno por letra** y vemos cuál explica mejor el trazo actual.

```mermaid
flowchart LR
    A[Trazo del usuario] --> B[Direcciones<br/>discretas]
    B --> C[27 modelos HMM<br/>uno por letra]
    C --> D[Probabilidad<br/>de cada letra]
    D --> E[Ranking + línea<br/>punteada]
```

---

## Cómo funciona

### 1. Convertir el trazo en símbolos
Cada tramo del trazo se convierte en un símbolo que combina dos cosas:

- **La dirección** del movimiento, en **16 sectores** (Chain Code, K = 16): 0 = →, 4 = ↑, 8 = ←, 12 = ↓ y las intermedias.
- **La zona vertical**: a qué altura va el trazo respecto del punto donde empezó la letra (0 = arriba, 1 = a la altura del inicio, 2 = abajo; el borde está a 4 pasos, 0.12 del lienzo).

`símbolo = dirección + 16 · zona`, así que el alfabeto tiene 16 × 3 = **48 símbolos**. La zona le dice al modelo dónde va el trazo: sin ella, una «h» y una «n», o una «b» y una «p», se mueven igual.

Antes de cuantizar, el trazo se suaviza y se remuestrea por **longitud de arco** (un punto cada 3 % del lienzo), de modo que la velocidad con la que se dibuja no cambia la secuencia. Detalles en [`docs/contrato_features.md`](docs/contrato_features.md).

### 2. Un modelo HMM por letra
Cada modelo está definido por λ = (A, B, π):

| Elemento | Qué significa |
|---|---|
| **A** | Probabilidad de pasar de una fase del trazo a la siguiente |
| **B** | Probabilidad de ver cada símbolo (dirección y zona) en cada fase |
| **π** | Probabilidad de empezar en cada fase |

```mermaid
flowchart LR
    q1((Inicio)) --> q2((Curva)) --> q3((Cierre))
```
$$\hat c = \arg\max_c P(O_{1:t} \mid \lambda_c)$$

### 3. Autocompletado
Con el modelo de la letra ganadora se toman las fases que faltan, se obtiene la dirección más probable de cada una y se dibuja ese camino desde el cursor.

---

## Pantallas

```mermaid
flowchart LR
    A[Menú principal] --> B[Ventana 2D]
    A --> C[Ventana 3D esfera]
```

Cada ventana tiene el **lienzo a la izquierda** y el **ranking de letras a la derecha**, con botones para limpiar y reiniciar.

La ventana 2D tiene:

- **Lienzo** con las cuatro líneas guía (ascendente, altura x, línea base y descendente), en las mismas proporciones que Glifo Capture.
- **Proyección punteada** de lo que le falta a la letra más probable, que se actualiza con cada símbolo nuevo. Se ajusta a lo que ya dibujaste: arranca en el cursor y termina donde le corresponde a TU letra (su tamaño y su línea base).
- **Ranking** de P(c | O₁:ₜ): las 4 letras más probables y el resto agrupado.
- **«Trazo no definido»** cuando el trazo no se parece a ninguna letra. En ese caso no se dibuja proyección.
- **«Guía de letras»**: anima cada letra trazo por trazo, en orden y con sus números, como la guía de Glifo Capture. Muestra la misma guía hecha a mano de Glifo Capture (`data/guia_dibujo.json`).

La ventana 3D tiene la misma estructura, con una **esfera** como lienzo:

- **Controles:** clic izquierdo dibuja, clic derecho (o el botón del medio) gira la esfera, la rueda acerca o aleja. «Centrar vista» la devuelve a como empezó, sin borrar.
- **«Arriba» está definido en la esfera: es su polo norte** (marcado con una N). No depende del panel: si la esfera se gira, «arriba» gira con ella. Las líneas guía son paralelos de la esfera, alrededor del ecuador, y una brújula en la esquina muestra hacia dónde queda el norte en pantalla.
- **Cuenta toda la esfera**, se vea o no. Lo que queda en la cara de atrás se sigue analizando y se deja ver, tenue, a través de la esfera. Girar la vista no cambia el resultado.
- Los puntos se pasan a **coordenadas esféricas (θ, φ)** y los símbolos salen de las variaciones (Δθ, Δφ) sobre la superficie (ver `docs/contrato_features.md`).
- La **proyección punteada** se dibuja pegada a la superficie y gira con la esfera.
- Usa **sus propios modelos**, entrenados con un dataset 3D sintético hecho a partir del 2D (ver «Modo 3D» más abajo).

Atajo: `Esc` vuelve al menú.

---

## Requisitos

| Qué | Detalle |
|---|---|
| **Python** | 3.9 o superior (probado con 3.13) |
| **Extractor, datos y pruebas** | Solo usan la librería estándar de Python |
| **HMMEngine** | Usa NumPy (ver `requirements.txt`) |
| **GUI** | PySide6 (Qt 6). Las fuentes van incluidas en `assets/fonts` |
| **Esfera 3D** | OpenGL 2.1 o superior, con las clases de Qt (no hay que instalar nada más). En Windows, si la tarjeta no tiene OpenGL, Qt usa su renderizador por software |

```bash
pip install -r requirements.txt
```

> En Windows, si `python` no responde, usa `py` en su lugar.

---

## Cómo ejecutar

Todos los comandos se lanzan **desde la raíz del proyecto**.

### La aplicación

```bash
python GUI.py
```

Al abrir el modo 2D se cargan los modelos de `data/modelos_letras_entrenados.json`. Los del modo 3D (`data/modelos_letras_3d.json`) se cargan la primera vez que se entra a la esfera; tarda unos segundos.

### Entrenar los modelos HMM

```bash
python training/entrenar_hmms.py
```

Entrena con Baum-Welch los 27 modelos a partir de `data/secuencias_2d_k16.json` y los guarda en `data/modelos_letras_entrenados.json`.

### Modo 3D: dataset sintético y modelos

Solo hay dataset 2D, así que el del modo 3D se fabrica a partir de él:

```bash
python training/build_sequences_3d.py          # pega cada muestra 2D sobre la esfera -> data/secuencias_3d_k16.json
python training/entrenar_hmms.py --modo 3d     # entrena los 27 modelos 3D -> data/modelos_letras_3d.json
```

Cada muestra se pega varias veces sobre la esfera, derecha respecto al polo norte, en cualquier longitud (también en la cara de atrás) y a una latitud de hasta ±0.6 rad del ecuador, con tamaño (±30 %) y giro (±8°) al azar, enrollando el lienzo sobre la superficie con el mapa exponencial. El reparto entrenamiento/prueba es **por muestra** y se hace antes de pegarlas: ninguna muestra de prueba aparece en el entrenamiento, ni siquiera en otro lugar de la esfera.

Acierto de los modelos 3D sobre la prueba 3D (996 secuencias): **96.5 %** solo con el HMM y **97.1 %** con los términos que usa la app al levantar el lápiz (ver «Cómo decide la letra»).

El acierto depende de la latitud, porque junto a los polos los meridianos se juntan y «arriba» deja de ser una sola dirección a lo ancho de la letra: es 97.8 % en el ecuador, 95.4 % a 0.6 rad (34°), 88 % a 0.9 rad (52°) y 75 % a 1.1 rad (63°).

### La guía de dibujo

```bash
python training/generar_guia.py                          # -> data/guia_trazos.json (plantillas del reconocedor)
```

Hay dos guías. La que **se muestra** en la «Guía de letras» es la de Glifo Capture, hecha a mano (`data/guia_dibujo.json`, copiada de su `docs/guia.js`): solo es visual. La que **usa el reconocedor** como plantilla para proyectar el trazo probable sale **del dataset**: cada trazo es el trazo mediano de todas las muestras de la letra (remuestreadas por longitud de arco), suavizado con curvas de Bézier. Así coincide con el diseño con el que aprenden los modelos. Con la guía hecha a mano como plantilla, que no coincidía con el dataset en la «x», la «k», la «d», la «p» o la «y», quien la seguía tenía 85.6 % de acierto; con la generada, 98.6 % en 2D y 99.5 % en 3D. **Volver a generarla cada vez que cambie el dataset** (`tests/test_guia.py` avisa si quedó desactualizada).

### Cómo decide la letra

Mientras se dibuja se compara la probabilidad de que el trazo sea el **comienzo** de cada letra. Al levantar el lápiz se suman dos términos (ver `HMMEngine.classify`), con pesos elegidos sobre las secuencias de **entrenamiento**:

| Término | Peso | Para qué |
|---|---|---|
| El modelo termina en su último estado, log α_T(N) | 0.2 | Que una letra terminada no se confunda con el comienzo de otra más larga |
| El largo de la secuencia, log N(T; media, desviación de la letra) | 2 | El HMM modela mal las duraciones: sin esto, la «n» absorbía el palito largo de la «h» |

Acierto sobre el 20 % de prueba (2D): **95.6 %** solo con el HMM y **97.8 %** con los dos términos.

Cómo se llegó ahí (mismas muestras de prueba):

| Símbolos | Solo HMM | Con los dos términos |
|---|---|---|
| 8 direcciones | 93.6 % | 95.8 % |
| 16 direcciones × 3 zonas (actual) | 95.6 % | 97.8 % |

La configuración se eligió con una validación apartada del entrenamiento, no con la prueba. Ahí también se probó, sin mejora: más estados por letra (12 y 16), 4 y 5 zonas, y entrenar con copias deformadas (aumento de datos).

### Pruebas

```bash
python -m unittest discover -s tests -v
```

### Revisar el dataset

```bash
python training/check_data.py
```

Muestra cuántas muestras hay por letra (la meta de Glifo Capture es 70; se cambia con `--meta`), cuántos símbolos genera cada una y avisa de trazos demasiado cortos, de longitud atípica o fuera de la escala esperada.

### Generar las secuencias de entrenamiento

```bash
python training/build_sequences.py                 # K=16 y 3 zonas, 20 % para prueba
python training/build_sequences.py --holdout 0     # todo a entrenamiento (modelo final)
```

Lee el `data/glifo-dataset-*.json` más reciente y crea `data/secuencias_2d_k16.json`, que consume el entrenamiento del HMM. Cada muestra del dataset es `letra → muestras → trazos → [x, y, t_ms]`, con `x`, `y` en la escala 0–1000 relativa al lienzo de Glifo Capture (no son píxeles).

> **Regenerar con `python training/build_sequences.py` cada vez que cambie el dataset.** `secuencias_2d_k16.json` es un archivo generado: si el dataset cambia y no se vuelve a crear, queda desactualizado sin que nadie lo note.

---

## Estructura del proyecto

```
Glifo/
├── GUI.py                        # Punto de entrada: python GUI.py
├── src/
│   ├── FeatureExtractor.py       # Chain Code, remuestreo, símbolos <-> puntos, coordenadas esféricas
│   ├── HMMEngine.py              # Forward, Backward, Baum-Welch, Viterbi, clasificación y proyección
│   └── gui/                      # Interfaz (PySide6)
│       ├── main_window.py        # Secciones: menú, 2D y 3D
│       ├── menu_view.py          # Ventana A: menú principal
│       ├── view_2d.py            # Ventana B: reconocedor 2D
│       ├── view_3d.py            # Ventana C: reconocedor 3D (esfera)
│       ├── canvas_2d.py          # Lienzo con líneas guía y proyección
│       ├── canvas_3d.py          # Esfera con OpenGL: dibujar, girar, acercar
│       ├── ranking_panel.py      # Listado de probabilidades
│       ├── guide_popup.py        # Guía de letras (animada, trazos de Glifo Capture)
│       ├── guia_trazos.py        # Lee las dos guías y las convierte en puntos
│       ├── recognizer.py         # Puente FeatureExtractor + HMMEngine
│       ├── recognizer_3d.py      # El mismo puente para la esfera (toda la esfera, norte en su polo)
│       ├── widgets.py            # Componentes visuales compartidos
│       └── theme.py              # Colores, fuentes y estilos
├── training/
│   ├── dataset.py                # Lector del JSON del dataset
│   ├── check_data.py             # Revisión de calidad de los datos
│   ├── build_sequences.py        # Dataset -> secuencias de símbolos
│   ├── build_sequences_3d.py     # Dataset 2D -> secuencias 3D sintéticas sobre la esfera
│   ├── generar_guia.py           # Guía de dibujo a partir del dataset (trazo mediano por letra)
│   └── entrenar_hmms.py          # Entrena y exporta los 27 modelos HMM (--modo 2d | 3d)
├── tests/
│   ├── test_features.py          # Pruebas del extractor
│   ├── test_features_3d.py       # Pruebas de la geometría esférica
│   ├── test_dataset.py           # Pruebas del lector y de build_sequences
│   ├── test_sequences_3d.py      # Pruebas del dataset 3D sintético
│   ├── test_recognizer.py        # Pruebas del puente GUI <-> HMM
│   ├── test_recognizer_3d.py     # Pruebas del reconocedor 3D
│   └── test_guia.py              # Quien sigue la guía es reconocido
├── docs/
│   └── contrato_features.md      # Contrato entre módulos
├── assets/fonts/                 # Selawik, Playwrite AR (a y q) e Inconsolata (OFL)
├── data/
│   ├── glifo-dataset-*.json      # Trazos de ejemplo
│   ├── secuencias_2d_k16.json     # Secuencias GENERADAS (regenerar si cambia el dataset)
│   ├── secuencias_3d_k16.json     # Secuencias 3D sintéticas GENERADAS
│   ├── guia_trazos.json          # Plantillas GENERADAS del dataset (reconocedor)
│   ├── guia_dibujo.json          # Guía que se muestra (la de Glifo Capture, hecha a mano)
│   ├── modelos_letras_entrenados.json  # Modelos HMM del modo 2D
│   └── modelos_letras_3d.json    # Modelos HMM del modo 3D
├── requirements.txt
├── .gitattributes                # Finales de línea LF
└── .gitignore
```
