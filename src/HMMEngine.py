import numpy as np
import json

class HMMEngine:
    def __init__(self):
        # El motor almacena los modelos en memoria
        self.modelos = {} 

    # ---------------------------------------------------------
    # 1. MATEMÁTICAS CORE (Algoritmos base)
    # ---------------------------------------------------------
    def _inicializar_hmm(self, N, M, secuencias=None, piso=1e-3):
        """Inicializa pi, A, B con topología estricta Left-to-Right.

        Sin `secuencias`, B arranca al azar. Con `secuencias` se usa la
        inicialización por tramos, la habitual en modelos izquierda-derecha:
        cada secuencia se parte en N tramos iguales, el tramo i se asigna al
        estado i y B arranca con la frecuencia de cada símbolo en su tramo
        (más un piso para que ninguna emisión empiece en 0). La
        autotransición a_ii se elige para que la duración media de cada
        estado, 1/(1 - a_ii), sea el largo medio de un tramo (T_medio / N).
        """
        pi = np.zeros(N)
        pi[0] = 1.0

        if secuencias:
            largo_medio = np.mean([len(O) for O in secuencias])
            a_ii = min(0.95, max(0.5, 1.0 - N / largo_medio))
        else:
            a_ii = 0.5
        A = np.zeros((N, N))
        for i in range(N):
            if i == N - 1:
                A[i, i] = 1.0
            else:
                A[i, i] = a_ii
                A[i, i+1] = 1.0 - a_ii

        if secuencias:
            B = np.full((N, M), piso)
            for O in secuencias:
                for estado, tramo in enumerate(np.array_split(np.asarray(O, dtype=int), N)):
                    for k in tramo:
                        B[estado, k] += 1.0
        else:
            B = np.random.rand(N, M)
        B = B / np.sum(B, axis=1, keepdims=True)

        return pi, A, B

    def _algoritmo_forward_escalado(self, pi, A, B, O):
        """Función forward con escalado para evitar underflow."""
        T = len(O)
        N = A.shape[0]
        alfa = np.zeros((T, N))
        c = np.zeros(T)
        
        alfa[0, :] = pi * B[:, O[0]]
        c[0] = 1.0 / (np.sum(alfa[0, :]) + 1e-300)
        alfa[0, :] = alfa[0, :] * c[0]
        
        for t in range(1, T):
            for j in range(N):
                alfa[t, j] = np.sum(alfa[t-1, :] * A[:, j]) * B[j, O[t]]
            c[t] = 1.0 / (np.sum(alfa[t, :]) + 1e-300)
            alfa[t, :] = alfa[t, :] * c[t]
            
        return alfa, c

    def _algoritmo_backward_escalado(self, A, B, O, c):
        """Método interno usado por Baum-Welch (Backward)"""
        T = len(O)
        N = A.shape[0]
        beta = np.zeros((T, N))
        
        beta[T-1, :] = 1.0 * c[T-1]
        
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(A[i, :] * B[:, O[t+1]] * beta[t+1, :])
            beta[t, :] = beta[t, :] * c[t]
            
        return beta
        
    def algoritmo_viterbi(self, pi, A, B, O):
        """Algoritmo Viterbi (útil para defensa en tablero)."""
        T = len(O)
        N = A.shape[0]
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        delta[0, :] = pi * B[:, O[0]]
        for t in range(1, T):
            for j in range(N):
                probabilidades = delta[t-1, :] * A[:, j]
                delta[t, j] = np.max(probabilidades) * B[j, O[t]]
                psi[t, j] = np.argmax(probabilidades)

        ruta_optima = np.zeros(T, dtype=int)
        ruta_optima[T-1] = np.argmax(delta[T-1, :])

        for t in range(T-2, -1, -1):
            ruta_optima[t] = psi[t+1, ruta_optima[t+1]]

        return delta, psi, ruta_optima

    def viterbi_log(self, pi, A, B, O, terminar_en_final=False):
        """Viterbi en logaritmos: δ_t(j) = max_i [δ_{t-1}(i) + log a_ij] + log b_j(O_t).

        Mismo algoritmo que `algoritmo_viterbi`, pero sumando logaritmos en vez
        de multiplicar probabilidades, así que no se vuelve 0 con trazos largos.
        Con `terminar_en_final=True` la ruta debe acabar en el último estado.
        Devuelve (ruta óptima de estados, log P de esa ruta).
        """
        O = np.asarray(O, dtype=int)
        T, N = len(O), A.shape[0]
        with np.errstate(divide="ignore"):
            lpi, lA, lB = np.log(pi), np.log(A), np.log(B)
        delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)
        delta[0] = lpi + lB[:, O[0]]
        for t in range(1, T):
            candidatos = delta[t - 1][:, None] + lA        # candidatos[i, j]
            psi[t] = np.argmax(candidatos, axis=0)
            delta[t] = candidatos[psi[t], np.arange(N)] + lB[:, O[t]]
        final = N - 1 if terminar_en_final else int(np.argmax(delta[T - 1]))
        ruta = np.zeros(T, dtype=int)
        ruta[T - 1] = final
        for t in range(T - 2, -1, -1):
            ruta[t] = psi[t + 1, ruta[t + 1]]
        return ruta, float(delta[T - 1, final])

    def iteracion_baum_welch_offline(self, pi, A, B, secuencias, N, M, piso=0.0):
        """Lógica de entrenamiento por lotes con escalado.

        `piso` > 0 impide que una emisión b_i(k) quede exactamente en 0: si
        quedara en 0, un solo símbolo no visto en el entrenamiento anularía
        P(O | λ) y descartaría la letra aunque el resto del trazo encaje.
        """
        pi_num = np.zeros(N)
        A_num = np.zeros((N, N))
        A_den = np.zeros(N)
        B_num = np.zeros((N, M))
        B_den = np.zeros(N)

        trazos_validos = 0

        for O in secuencias:
            O = np.array(O, dtype=int)
            T = len(O)
            if T < N: 
                continue
            
            trazos_validos += 1
            
            alfa, c = self._algoritmo_forward_escalado(pi, A, B, O)
            beta = self._algoritmo_backward_escalado(A, B, O, c)

            gamma = np.zeros((T, N))
            xi = np.zeros((T-1, N, N))

            for t in range(T):
                gamma[t, :] = alfa[t, :] * beta[t, :]
                gamma[t, :] = gamma[t, :] / (np.sum(gamma[t, :]) + 1e-300)

            for t in range(T-1):
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = alfa[t, i] * A[i, j] * B[j, O[t+1]] * beta[t+1, j]
                xi[t, :, :] = xi[t, :, :] / (np.sum(xi[t, :, :]) + 1e-300)

            pi_num += gamma[0, :]
            for i in range(N):
                A_den[i] += np.sum(gamma[:-1, i])
                B_den[i] += np.sum(gamma[:, i])
                for j in range(N):
                    A_num[i, j] += np.sum(xi[:, i, j])
                for k in range(M):
                    B_num[i, k] += np.sum(gamma[O == k, i])

        if trazos_validos == 0:
            return pi, A, B

        pi_new = pi_num / np.sum(pi_num)
        
        # Un estado que no recibió ninguna probabilidad en esta iteración
        # (denominador 0) conserva sus parámetros anteriores; si no, su fila
        # quedaría en 0/0 y llenaría el modelo de NaN.
        A_new = A.copy()
        for i in range(N):
            if A_den[i] > 1e-12 and np.sum(A_num[i, :]) > 1e-12:
                A_new[i, :] = A_num[i, :] / A_den[i]

        B_new = B.copy()
        for i in range(N):
            if B_den[i] > 1e-12:
                B_new[i, :] = B_num[i, :] / B_den[i]

        A_new = np.triu(A_new)
        for i in range(N):
            A_new[i, i+2:] = 0.0
        filas = np.sum(A_new, axis=1, keepdims=True)
        A_new = np.where(filas > 0, A_new / np.where(filas > 0, filas, 1.0), A)

        if piso > 0:
            B_new = np.maximum(B_new, piso)
            B_new = B_new / np.sum(B_new, axis=1, keepdims=True)

        return pi_new, A_new, B_new

    # ---------------------------------------------------------
    # 2. ENTRENAMIENTO E I/O (Entrada/Salida)
    # ---------------------------------------------------------
    def entrenar_modelos(self, ruta_json_secuencias, estados_por_letra, M=8, epocas=5,
                         semilla=0, piso_emision=1e-3, init_por_tramos=True):
        """Llama a inicializar_hmm y baum_welch, y guarda el resultado en self.modelos.

        `semilla` fija el azar para que entrenar dos veces dé los mismos modelos.
        `piso_emision` es el mínimo de cada b_i(k) en cada iteración.
        `init_por_tramos` usa la inicialización por tramos en lugar de B al azar.
        """
        np.random.seed(semilla)
        with open(ruta_json_secuencias, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        secuencias_por_letra = datos.get('secuencias', {})

        for letra, secuencias in secuencias_por_letra.items():
            N = estados_por_letra.get(letra, 4)
            pi, A, B = self._inicializar_hmm(N, M, secuencias if init_por_tramos else None, piso_emision)

            print(f"Entrenando letra '{letra}' ({N} estados) - {len(secuencias)} secuencias...")

            for epoca in range(epocas):
                pi, A, B = self.iteracion_baum_welch_offline(pi, A, B, secuencias, N, M, piso_emision)
            
            self.modelos[letra] = {
                'N': N,
                'pi': pi,
                'A': A,
                'B': B
            }

    def exportar_modelos(self, ruta_salida):
        """Guarda self.modelos en un archivo .json."""
        if not self.modelos:
            print("No hay modelos para exportar.")
            return

        modelos_exportables = {}
        for letra, params in self.modelos.items():
            modelos_exportables[letra] = {
                'N': params['N'],
                'pi': params['pi'].tolist() if isinstance(params['pi'], np.ndarray) else params['pi'],
                'A': params['A'].tolist() if isinstance(params['A'], np.ndarray) else params['A'],
                'B': params['B'].tolist() if isinstance(params['B'], np.ndarray) else params['B']
            }
            
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(modelos_exportables, f, indent=4)
        print(f"Modelos guardados en {ruta_salida}")
        
    def cargar_modelos(self, ruta_json):
        """Lee un .json y lo carga en la variable self.modelos"""
        with open(ruta_json, 'r') as f:
            datos = json.load(f)
            for letra, params in datos.items():
                self.modelos[letra] = {
                    'pi': np.array(params['pi']),
                    'A': np.array(params['A']),
                    'B': np.array(params['B']),
                    'N': params['N']
                }

    # ---------------------------------------------------------
    # 3. MÉTODOS DE INFERENCIA (Los que usa la GUI)
    # ---------------------------------------------------------
    def classify(self, O, prior_uniforme=True, temperatura=1.0, completa=False):
        """Clasifica una secuencia (completa o parcial).

        Con `completa=False` usa P(O_1:t | λ_c) = Σ_i α_t(i): la probabilidad de
        que el trazo sea el COMIENZO de la letra (sirve mientras se dibuja).
        Con `completa=True` usa P(O, q_T = N | λ_c) = α_T(N): exige que el modelo
        termine en su último estado. Así una letra terminada no se confunde con
        el comienzo de otra más larga (la «i» con la «f», la «o» con la «a»).

        P(c | O) ∝ P(O | λ_c)^(1/T), con prior uniforme. Con T = 1 es la
        regla de Bayes tal cual. Como P(O | λ_c) multiplica una probabilidad
        por símbolo, en 20-30 símbolos cualquier diferencia se amplifica y casi
        siempre una letra se queda con ~100 % aunque esté equivocada. T > 1
        (escalado de temperatura) suaviza esos porcentajes para que la
        confianza mostrada se parezca al acierto real. No cambia el orden del
        ranking ni la letra ganadora.
        """
        if temperatura <= 0:
            raise ValueError("temperatura debe ser positiva.")
        if not self.modelos:
            raise ValueError("No hay modelos cargados.")
            
        probabilidades_log = {}
        
        for letra, modelo in self.modelos.items():
            alfa, c = self._algoritmo_forward_escalado(modelo['pi'], modelo['A'], modelo['B'], O)
            log_prob = -np.sum(np.log(c + 1e-300))
            if completa:
                # α escalado suma 1 en cada t: log α_T(N) = log α̂_T(N) + log P(O | λ).
                log_prob += np.log(alfa[-1, -1] + 1e-300)
            probabilidades_log[letra] = log_prob

        letras = list(probabilidades_log.keys())
        log_probs_array = np.array(list(probabilidades_log.values())) / temperatura

        log_probs_array -= np.max(log_probs_array)
        probs = np.exp(log_probs_array)
        
        suma_probs = np.sum(probs)
        probs_normalizadas = probs / (suma_probs + 1e-300)
        
        resultados = {letras[i]: probs_normalizadas[i] for i in range(len(letras))}
        
        letra_ganadora = max(resultados, key=resultados.get)
        return letra_ganadora, resultados
        
    @staticmethod
    def duraciones_esperadas(A, maximo=20):
        """Duración esperada (en símbolos) de cada estado: d_i = 1 / (1 - a_ii).

        En un modelo izquierda-derecha la permanencia en un estado es geométrica,
        así que su media es 1/(1 - a_ii). El último estado es absorbente
        (a_NN = 1, duración infinita): se le asigna la media de los demás.
        """
        diag = np.diag(A)
        d = np.zeros(len(diag))
        for i, a in enumerate(diag):
            d[i] = maximo if a >= 1.0 - 1e-9 else min(maximo, 1.0 / (1.0 - a))
        if len(d) > 1 and diag[-1] >= 1.0 - 1e-9:
            d[-1] = np.mean(d[:-1])
        return d

    def project(self, O, pasos_a_proyectar=None, letra=None):
        """Proyecta los símbolos restantes {O_t+1, ..., O_T} de la letra más probable.

        1. ĉ = argmax_c P(O_1:t | λ_c)  (o `letra`, si la GUI ya la conoce).
        2. Estado actual: argmax_i α_t(i) en el modelo de ĉ.
        3. Se recorre la trayectoria aprendida: en cada estado, desde el actual
           hasta el último, se emite el símbolo más probable argmax_k b_i(k)
           durante su duración esperada. Del estado actual solo se proyecta la
           mitad de su duración, porque parte ya se dibujó.

        `pasos_a_proyectar` limita la cantidad de símbolos (None = hasta el final).
        Devuelve (letra, lista de símbolos).
        """
        letra_predicha = letra if letra is not None else self.classify(O)[0]
        modelo = self.modelos[letra_predicha]
        A, B = modelo['A'], modelo['B']

        alfa, _ = self._algoritmo_forward_escalado(modelo['pi'], A, B, O)
        estado_actual = int(np.argmax(alfa[-1, :]))
        duraciones = self.duraciones_esperadas(A)

        emisiones_proyectadas = []
        for estado in range(estado_actual, A.shape[0]):
            pasos = duraciones[estado] / 2.0 if estado == estado_actual else duraciones[estado]
            simbolo = int(np.argmax(B[estado, :]))
            emisiones_proyectadas.extend([simbolo] * int(round(pasos)))

        if pasos_a_proyectar is not None:
            emisiones_proyectadas = emisiones_proyectadas[:pasos_a_proyectar]
        return letra_predicha, emisiones_proyectadas