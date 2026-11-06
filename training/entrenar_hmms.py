import sys
from pathlib import Path

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
    
    # 2. Definir el diccionario estático de estados por letra (tu topología)
    diccionario_estados = {
        'c': 3, 'e': 3, 'l': 3, 'o': 3, 's': 3, 'u': 3, 'v': 3, 'w': 4, 'z': 3,
        'a': 4, 'b': 5, 'd': 5, 'g': 5, 'h': 5, 'i': 5, 'j': 5, 'n': 5, 
        'p': 5, 'q': 6, 'r': 5, 't': 5, 'x': 5, 'y': 5,
        'f': 6, 'k': 6, 'm': 6
    }
    
    # 3. Configuración de archivos
    archivo_entrada = ROOT / 'data' / 'secuencias_2d_k8.json'
    archivo_salida = ROOT / 'data' / 'modelos_letras_entrenados.json'
    
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