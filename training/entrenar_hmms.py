import json
import sys
from pathlib import Path

import numpy as np

# Permite ejecutarlo desde la raíz del proyecto: python training/entrenar_hmms.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.HMMEngine import HMMEngine  # noqa: E402

def main():
    msg = "Iniciando el pipeline de entrenamiento de HMM..."
    print(msg)
    
    # 1. Instanciar el motor estocástico
    motor = HMMEngine()
    
    # 3. Configuración de archivos
    archivo_entrada = ROOT / 'data' / 'secuencias_2d_k8.json'
    archivo_salida = ROOT / 'data' / 'modelos_letras_entrenados.json'

    # 2. Estados por letra: 8, salvo en letras de secuencias cortas.
    #    Con 3 a 6 estados las letras quedaban poco detalladas («o», «c» y «s» tenían 3)
    #    y se confundían entre sí; con 8, inicialización por tramos y piso en B, el
    #    acierto sobre el 20 % de prueba subió de 66.5 % a 84.9 %.
    #    Baum-Welch descarta las secuencias más cortas que el número de estados, así que
    #    una letra corta (la «i» tiene ~4 símbolos) no se entrenaría con 8: en ese caso se
    #    usa el largo que alcanza el 90 % de sus secuencias (percentil 10), mínimo 3.
    ESTADOS = 8
    with open(archivo_entrada, encoding='utf-8') as f:
        secuencias = json.load(f)['secuencias']
    diccionario_estados = {
        letra: int(min(ESTADOS, max(3, np.percentile([len(O) for O in seqs], 10))))
        for letra, seqs in secuencias.items() if seqs
    }
    cortas = {l: n for l, n in diccionario_estados.items() if n < ESTADOS}
    if cortas:
        print(f"Letras con menos de {ESTADOS} estados por ser cortas: {cortas}")
    
    # 4. Entrenar los modelos
    print(f"Cargando observaciones desde: {archivo_entrada}")
    # Usamos M=8 (k8) y 15 épocas suelen ser un buen estándar para asegurar la convergencia en Baum-Welch
    motor.entrenar_modelos(ruta_json_secuencias=archivo_entrada, 
                           estados_por_letra=diccionario_estados, 
                           M=8, 
                           epocas=15)
    
    # 5. Exportar las matrices entrenadas
    motor.exportar_modelos(ruta_salida=archivo_salida)
    
    print("\n¡Proceso finalizado exitosamente!")
    print(f"Puedes entregar tu archivo '{archivo_salida}' al equipo.")

if __name__ == '__main__':
    main()