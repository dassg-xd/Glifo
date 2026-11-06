# Glifo

**Reconocedor de letras manuscritas en tiempo real (2D y 3D) con Cadenas Ocultas de Markov.**

Proyecto Final · Procesos Estocásticos (603504) · Universidad de los Llanos · Ingeniería de Sistemas

> **Estado:** la primera entrega cubre el modo **2D**. El modo 3D (esfera) llega en la segunda entrega.

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
Cada tramo del trazo se convierte en una de **8 direcciones** (Chain Code, K = 8):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| → | ↗ | ↑ | ↖ | ← | ↙ | ↓ | ↘ |

Antes de cuantizar, el trazo se suaviza y se remuestrea por **longitud de arco** (un punto cada 3 % del lienzo), de modo que la velocidad con la que se dibuja no cambia la secuencia. Detalles en [`docs/contrato_features.md`](docs/contrato_features.md).

### 2. Un modelo HMM por letra
Cada modelo está definido por λ = (A, B, π):

| Elemento | Qué significa |
|---|---|
| **A** | Probabilidad de pasar de una fase del trazo a la siguiente |
| **B** | Probabilidad de ver cada dirección en cada fase |
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

En esta entrega **solo funciona el modo 2D**. El botón «Esfera 3D» abre una vista vacía, con la barra superior para volver.

La ventana 2D tiene:

- **Lienzo** con las cuatro líneas guía (ascendente, altura x, línea base y descendente), en las mismas proporciones que Glifo Capture.
- **Proyección punteada** de la letra más probable, que se actualiza con cada símbolo nuevo.
- **Ranking** de P(c | O₁:ₜ): las 4 letras más probables y el resto agrupado.
- **«Trazo no definido»** cuando el trazo no se parece a ninguna letra. En ese caso no se dibuja proyección.
- **«Guía de letras»**: muestra cómo se dibuja cada letra y en cuántos trazos, según el dataset.

Atajo: `Esc` vuelve al menú.

---

## Requisitos

| Qué | Detalle |
|---|---|
| **Python** | 3.9 o superior (probado con 3.13) |
| **Extractor, datos y pruebas** | Solo usan la librería estándar de Python |
| **HMMEngine** | Usa NumPy (ver `requirements.txt`) |
| **GUI** | PySide6 (Qt 6). Las fuentes van incluidas en `assets/fonts` |

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

Al abrir el modo 2D se cargan los modelos de `data/modelos_letras_entrenados.json`.

### Entrenar los modelos HMM

```bash
python training/entrenar_hmms.py
```

Entrena con Baum-Welch los 27 modelos a partir de `data/secuencias_2d_k8.json` y los guarda en `data/modelos_letras_entrenados.json`.

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
python training/build_sequences.py                 # K=8, 20 % para prueba
python training/build_sequences.py --holdout 0     # todo a entrenamiento (modelo final)
```

Lee el `data/glifo-dataset-*.json` más reciente y crea `data/secuencias_2d_k8.json`, que consume el entrenamiento del HMM. Cada muestra del dataset es `letra → muestras → trazos → [x, y, t_ms]`, con `x`, `y` en la escala 0–1000 relativa al lienzo de Glifo Capture (no son píxeles).

> **Regenerar con `python training/build_sequences.py` cada vez que cambie el dataset.** `secuencias_2d_k8.json` es un archivo generado: si el dataset cambia y no se vuelve a crear, queda desactualizado sin que nadie lo note.

---

## Estructura del proyecto

```
Glifo/
├── GUI.py                        # Punto de entrada: python GUI.py
├── src/
│   ├── FeatureExtractor.py       # Chain Code, remuestreo, símbolos <-> puntos
│   ├── HMMEngine.py              # Forward, Backward, Baum-Welch, Viterbi, clasificación y proyección
│   └── gui/                      # Interfaz (PySide6)
│       ├── main_window.py        # Secciones: menú, 2D y 3D
│       ├── menu_view.py          # Ventana A: menú principal
│       ├── view_2d.py            # Ventana B: reconocedor 2D
│       ├── view_3d.py            # Ventana C: 3D (vacía en esta entrega)
│       ├── canvas_2d.py          # Lienzo con líneas guía y proyección
│       ├── ranking_panel.py      # Listado de probabilidades
│       ├── guide_popup.py        # Guía de letras
│       ├── recognizer.py         # Puente FeatureExtractor + HMMEngine
│       ├── widgets.py            # Componentes visuales compartidos
│       └── theme.py              # Colores, fuentes y estilos
├── training/
│   ├── dataset.py                # Lector del JSON del dataset
│   ├── check_data.py             # Revisión de calidad de los datos
│   ├── build_sequences.py        # Dataset -> secuencias de símbolos
│   └── entrenar_hmms.py          # Entrena y exporta los 27 modelos HMM
├── tests/
│   ├── test_features.py          # Pruebas del extractor
│   ├── test_dataset.py           # Pruebas del lector y de build_sequences
│   └── test_recognizer.py        # Pruebas del puente GUI <-> HMM
├── docs/
│   └── contrato_features.md      # Contrato entre módulos
├── assets/fonts/                 # Selawik, Playwrite AR (a y q) e Inconsolata (OFL)
├── data/
│   ├── glifo-dataset-*.json      # Trazos de ejemplo
│   ├── secuencias_2d_k8.json     # Secuencias GENERADAS (regenerar si cambia el dataset)
│   └── modelos_letras_entrenados.json  # Modelos HMM que carga la GUI
├── requirements.txt
├── .gitattributes                # Finales de línea LF
└── .gitignore
```
