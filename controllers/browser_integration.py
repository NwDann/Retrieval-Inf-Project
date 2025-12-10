from typing import List, Tuple
from pathlib import Path
import numpy as np
from math import log2
# Importaciones relativas a los archivos que ya hemos creado/discutido
from .loadmodel import cargarModelo
from .corpus_loader import obtenerCorpus
import logging

# Configurar logging
# La ruta del archivo de log se resuelve dos niveles arriba (proyecto raíz)
rutaArchivoLog = Path(__file__).resolve().parents[1] / "debug.log"
logging.basicConfig(
    filename=str(rutaArchivoLog),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CalculadorMetricas:
    """Calcula las métricas de Precisión, Exhaustividad (Recall) y MAP."""

    @staticmethod
    def calcularPrecisionK(documentosRecuperados: List[int], documentosRelevantes: List[int], k: int) -> float:
        """ Calcula Precisión @ k (P@k) """
        if not documentosRecuperados:
            return 0.0
        
        recuperadosK = set(documentosRecuperados[:k])
        relevantes = set(documentosRelevantes)
        
        # Número de documentos relevantes recuperados entre los top K
        relevantesRecuperados = len(recuperadosK.intersection(relevantes))
        
        return relevantesRecuperados / k

    @staticmethod
    def calcularRecallK(documentosRecuperados: List[int], documentosRelevantes: List[int], k: int) -> float:
        """ Calcula Exhaustividad @ k (R@k) """
        if not documentosRelevantes:
            return 0.0
        
        recuperadosK = set(documentosRecuperados[:k])
        relevantes = set(documentosRelevantes)
        
        # Número de documentos relevantes recuperados entre los top K
        relevantesRecuperados = len(recuperadosK.intersection(relevantes))
        
        return relevantesRecuperados / len(relevantes)
    
    @staticmethod
    def calcularMAP(documentosRecuperados: List[int], documentosRelevantes: List[int]) -> float:
        """ Calcula Precisión Media Promedio (MAP) """
        precisiones = []
        numRelevantesEncontrados = 0
        
        relevantes = set(documentosRelevantes)
        
        for i, idDoc in enumerate(documentosRecuperados):
            if idDoc in relevantes:
                numRelevantesEncontrados += 1
                # Precision en la posición i+1
                precisionEnK = numRelevantesEncontrados / (i + 1)
                precisiones.append(precisionEnK)

        if not documentosRelevantes or not precisiones:
            return 0.0
        
        # MAP es la suma de las precisiones en cada posición relevante, dividida por el total de relevantes
        return sum(precisiones) / len(documentosRelevantes)

class NavegadorModelos:
    """Clase puente simple para cargar un modelo serializado (.pkl) y ejecutar búsquedas.

    La clase es intencionalmente mínima: intenta llamar al método `buscar` del modelo.
    Soporta modelos que retornan:
    1. Una lista de tuplas (id_doc, score) (TF-IDF / BM25).
    2. Un array/lista de índices de documentos coincidentes (Modelo Binario).
    """

    def __init__(self) -> None:
        """Inicializa el navegador de modelos."""
        self.modelo = None
        self.rutaModelo = None

        # Resolver la raíz del proyecto (dos niveles arriba de controllers/)
        self.raizProyecto = Path(__file__).resolve().parents[1]

    def cargar(self, ruta: str) -> Tuple[bool, str]:
        """Carga un archivo pickle usando la función global cargarModelo.
        
        Args:
            ruta: Ruta del archivo .pkl a cargar.
            
        Retorna:
            Tuple[bool, str]: (éxito, mensaje de estado).
        """
        # Convertir a ruta absoluta
        rutaAbsoluta = Path(ruta).resolve()
        
        # Verificar que el archivo existe
        if not rutaAbsoluta.exists():
            mensaje = f"El archivo no existe: {rutaAbsoluta}"
            logger.error(mensaje)
            return False, mensaje
        
        modelo = cargarModelo(str(rutaAbsoluta))
        if modelo is None:
            mensaje = f"No se pudo cargar el modelo desde: {rutaAbsoluta}"
            logger.error(mensaje)
            return False, mensaje

        self.modelo = modelo
        self.rutaModelo = str(rutaAbsoluta)
        
        nombreClase = type(modelo).__name__
        mensaje = f"Modelo cargado: {nombreClase}"
        logger.info(mensaje)
        return True, mensaje

    def tieneModelo(self) -> bool:
        """Verifica si un modelo ha sido cargado."""
        return self.modelo is not None

    def listarModelos(self, directorioModelos: str = "models") -> List[Path]:
        """
        Retorna una lista de objetos Path para los archivos .pkl en el directorio 'models' del proyecto.
        """
        carpeta = (self.raizProyecto / directorioModelos).resolve()
        if not carpeta.exists() or not carpeta.is_dir():
            logger.warning(f"Directorio de modelos no encontrado: {carpeta}")
            return []

        archivosPkl = sorted(carpeta.glob("*.pkl"))
        return archivosPkl

    def obtenerRutaModelo(self, tipoModelo: str, directorioModelos: str = "models") -> str:
        """Retorna la ruta absoluta del modelo según el tipo (binary, tfidf, bm25).
        
        Retorna la ruta absoluta como string, o string vacío si no lo encuentra.
        """
        carpeta = (self.raizProyecto / directorioModelos).resolve()
        if not carpeta.exists() or not carpeta.is_dir():
            return ""

        # Mapeo directo de nombres de archivos (exactos)
        mapaNombresArchivo = {
            'binary': 'modeloBinario.pkl',
            'tfidf': 'modeloTfIdf.pkl',
            'bm25': 'modeloBM25.pkl'
        }

        tipoModeloMin = tipoModelo.lower()
        nombreArchivo = mapaNombresArchivo.get(tipoModeloMin)
        
        if nombreArchivo:
            rutaCompleta = carpeta / nombreArchivo
            if rutaCompleta.exists():
                return str(rutaCompleta)
        
        return ""

    def buscar(self, consulta: str, k: int = 5) -> List[str]:
        """Ejecuta una búsqueda contra el modelo cargado y retorna strings formateados.
        
        Args:
            consulta: La consulta del usuario.
            k: Número máximo de resultados a retornar (límite).
            
        Retorna:
            List[str]: Lista de líneas legibles para mostrar en la UI.
        """
        logger.debug(f"Iniciando búsqueda con consulta: '{consulta}' y k={k}")
        
        if not self.modelo:
            logger.error("No hay modelo cargado")
            return []

        nombreModelo = type(self.modelo).__name__
        logger.debug(f"Modelo en uso: {nombreModelo}")
        
        resultado = None
        
        try:
            # Dado que hemos modificado todos los modelos para aceptar 'k',
            # la llamada es uniforme, lo cual simplifica la lógica.
            logger.debug(f"Llamando a {nombreModelo}.buscar('{consulta}', k={k})")
            resultado = self.modelo.buscar(consulta, k)
            
            logger.debug(f"Resultado obtenido: tipo={type(resultado)}, len={len(resultado) if hasattr(resultado, '__len__') else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Error al ejecutar la búsqueda en {nombreModelo}: {e}", exc_info=True)
            return []

        if resultado is None or len(resultado) == 0:
            return ["No se encontraron resultados relevantes."]

        # Obtener los IDs de documentos recuperados (limpios de scores)
        # Esto unifica la forma en que manejamos los resultados de los 3 modelos.
        idDocumentosRecuperados = []
        
        if nombreModelo == 'ModeloBinario':
            # El Modelo Binario devuelve directamente una lista de IDs (ya limitada a k)
            idDocumentosRecuperados = [int(i) for i in resultado if isinstance(i, (int, np.integer, float))]
        
        elif isinstance(resultado, list) and len(resultado) > 0 and isinstance(resultado[0], (tuple, list)):
            # TF-IDF / BM25 devuelven una lista de tuplas (id_doc, score)
            idDocumentosRecuperados = [int(item[0]) for item in resultado]

        # Aplicar límite K a los IDs (aunque ya debería estar aplicado en la llamada)
        idDocumentosRecuperados = idDocumentosRecuperados[:k]

        # ----------------------------------------------------
        # --- Lógica de Qrels y Métricas ---
        # ----------------------------------------------------
        corpus = obtenerCorpus()
        documentosRelevantes = corpus.obtenerQrels(consulta)
        esQrel = len(documentosRelevantes) > 0
        
        if esQrel:
            # Calcular las métricas
            pK = CalculadorMetricas.calcularPrecisionK(idDocumentosRecuperados, documentosRelevantes, k)
            rK = CalculadorMetricas.calcularRecallK(idDocumentosRecuperados, documentosRelevantes, k)
            mapScore = CalculadorMetricas.calcularMAP(idDocumentosRecuperados, documentosRelevantes)
            
            # Formato de la primera línea con las métricas
            lineasFormateadas = [
                f"✅ Qrel Encontrado: {consulta}",
                f"📊 Métricas (k={k}): P@{k}={pK:.3f} | R@{k}={rK:.3f} | MAP={mapScore:.3f}"
            ]
        else:
            # Si no es Qrel, solo se muestra la pregunta original (lo que el usuario tipeó)
            lineasFormateadas = [f"🔍 Búsqueda: {consulta}"]
        
        
        # ----------------------------------------------------
        # --- Formateo de Resultados ---
        # ----------------------------------------------------

        # Recorrer los resultados recuperados (IDs y Scores/Nones)
        for i, idDoc in enumerate(idDocumentosRecuperados):
            vistaPrevia = corpus.obtenerVistaPreviaDocumento(idDoc, maxCaracteres=50)
            score = None
            
            # Intentar obtener el score si existe (TF-IDF/BM25)
            if nombreModelo != 'ModeloBinario' and i < len(resultado):
                 try:
                    score = resultado[i][1]
                 except IndexError:
                    pass
            
            # Formato de línea única para todos
            if score is not None:
                linea = f"Doc {idDoc} — Score: {float(score):.4f} | {vistaPrevia}"
            else:
                linea = f"Doc {idDoc}: {vistaPrevia}"

            lineasFormateadas.append(linea)

        logger.debug(f"{len(lineasFormateadas)} líneas finales formateadas.")
        return lineasFormateadas


# Instancia global del navegador
_instanciaNavegador = None


def obtenerNavegador() -> NavegadorModelos:
    """Obtiene o crea la instancia global del navegador."""
    global _instanciaNavegador
    if _instanciaNavegador is None:
        _instanciaNavegador = NavegadorModelos()
    return _instanciaNavegador