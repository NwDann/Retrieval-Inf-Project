import pickle
import sys
from pathlib import Path

# Obtener la raíz del proyecto (dos niveles arriba de este archivo)
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Importar las clases necesarias para deserializar los modelos
from classes.binarymodel import ModeloBinario
from classes.tfidfmodel import ModeloVectorialTfIdf
from classes.bm25model import ModeloBM25


class ModuleMapper(pickle.Unpickler):
    """Mapea módulos __main__ a los módulos correctos durante la deserialización de pickle."""
    
    def find_class(self, module, name):
        """Intercepta la búsqueda de clases en pickle."""
        # Si el módulo es __main__, mapear a los módulos correctos
        if module == '__main__':
            if name == 'ModeloBinario':
                return ModeloBinario
            elif name == 'ModeloVectorialTfIdf':
                return ModeloVectorialTfIdf
            elif name == 'ModeloBM25':
                return ModeloBM25
        
        # En caso contrario, usar el comportamiento normal
        return super().find_class(module, name)


def cargarModelo(nombreArchivo):
    """Carga el objeto del modelo guardado usando pickle.
    
    Usa un mapeo especial para modelos que fueron pickleados con __main__.
    """
    try:
        with open(nombreArchivo, 'rb') as archivoEntrada:
            # Usar nuestro unpickler personalizado
            unpickler = ModuleMapper(archivoEntrada)
            modeloCargado = unpickler.load()
            
        print(f"\n✓ Modelo cargado exitosamente desde: {nombreArchivo}")
        return modeloCargado
    except FileNotFoundError:
        print(f"\n✗ Error: Archivo no encontrado: {nombreArchivo}")
        return None
    except Exception as e:
        print(f"✗ Error al cargar el modelo: {e}")
        import traceback
        traceback.print_exc()
        return None