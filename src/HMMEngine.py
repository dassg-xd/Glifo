import numpy as np
import json

class HMMEngine:
    def __init__(self):
        # El motor almacena los modelos en memoria
        self.modelos = {} 

    # ---------------------------------------------------------
    # 1. MATEMÁTICAS CORE (Algoritmos base)
    # ---------------------------------------------------------
    def _inicializar_hmm(self, N, M):
        """Inicializa pi, A, B con topología estricta Left-to-Right"""
        pi = np.zeros(N)
        pi[0] = 1.0

        A = np.zeros((N, N))
        for i in range(N):
            if i == N - 1:
                A[i, i] = 1.0 
            else:
                A[i, i] = 0.5
                A[i, i+1] = 0.5

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

    def iteracion_baum_welch_offline(self, pi, A, B, secuencias, N, M):
        """Lógica de entrenamiento por lotes con escalado."""
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
        
        A_new = np.zeros((N, N))
        for i in range(N):
            if A_den[i] > 0:
                A_new[i, :] = A_num[i, :] / A_den[i]
                
        B_new = np.zeros((N, M))
        for i in range(N):
            if B_den[i] > 0:
                B_new[i, :] = B_num[i, :] / B_den[i]

        A_new = np.triu(A_new)
        for i in range(N):
            A_new[i, i+2:] = 0.0
        A_new = A_new / np.sum(A_new, axis=1, keepdims=True)

        return pi_new, A_new, B_new

    # ---------------------------------------------------------
    # 2. ENTRENAMIENTO E I/O (Entrada/Salida)
    # ---------------------------------------------------------
    def entrenar_modelos(self, ruta_json_secuencias, estados_por_letra, M=8, epocas=5):
        """Llama a inicializar_hmm y baum_welch, y guarda el resultado en self.modelos"""
        with open(ruta_json_secuencias, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        secuencias_por_letra = datos.get('secuencias', {})

        for letra, secuencias in secuencias_por_letra.items():
            N = estados_por_letra.get(letra, 4) 
            pi, A, B = self._inicializar_hmm(N, M)
            
            print(f"Entrenando letra '{letra}' ({N} estados) - {len(secuencias)} secuencias...")
            
            for epoca in range(epocas):
                pi, A, B = self.iteracion_baum_welch_offline(pi, A, B, secuencias, N, M)
            
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
    def classify(self, O, prior_uniforme=True):
        """Clasifica una secuencia (completa o parcial)."""
        if not self.modelos:
            raise ValueError("No hay modelos cargados.")
            
        probabilidades_log = {}
        
        for letra, modelo in self.modelos.items():
            _, c = self._algoritmo_forward_escalado(modelo['pi'], modelo['A'], modelo['B'], O)
            log_prob = -np.sum(np.log(c + 1e-300))
            probabilidades_log[letra] = log_prob

        letras = list(probabilidades_log.keys())
        log_probs_array = np.array(list(probabilidades_log.values()))
        
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