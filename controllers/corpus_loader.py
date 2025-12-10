import pandas as pd
from typing import List
import logging
from pathlib import Path

# Configuración del logger
logger = logging.getLogger(__name__)

QRELS_PRECALCULADOS = {
    "What is (are) Parkinson's Disease": [7894, 7899, 7903, 7904, 7906, 7907, 7910, 7912, 7913],
    "What is (are) Colorectal Cancer": [151, 209, 7513, 7519, 7520, 7524, 7526, 7527, 7528, 7529, 7532],
    "What is (are) High Blood Pressure": [8000, 8005, 8006, 8007, 8008, 8015, 8016, 8017, 8018],
    "What causes Causes of Diabetes": [6583, 6584, 6585, 6586, 6587, 6633, 6634, 6635, 6636, 6637, 7069, 7070, 7071, 7072, 7073, 7230, 7231, 7232, 7233, 7234],
    "What is (are) Nutrition for Advanced Chronic Kidney Disease in Adults": [6257, 6258, 6259, 6260, 6261, 6262, 6263, 6264, 6265, 6267],
    "What is (are) Stroke": [7720, 7725, 7727, 7729, 7731, 7734, 7735, 7736, 7741, 7742],
    "What are the treatments for Breast Cancer": [664, 695, 7918, 7919, 7932, 7934, 7935, 7937, 7938, 7939, 7940, 7941],
    "What is (are) Skin Cancer": [112, 205, 232, 7592, 7598, 7599, 7600, 7601, 7602, 7603, 7606, 7608],
    "What are the treatments for Prostate Cancer": [101, 7656, 7657, 7671, 7672, 7673, 7675, 7676, 7677, 7678],
    "Who is at risk for Prostate Cancer": [54, 55, 82, 84, 7654, 7664, 7665, 7666, 7679],
    "What is (are) Age-related Macular Degeneration": [7328, 7332, 7333, 7334, 7335, 7336, 7342, 7343, 7347],
    "What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis": [7087, 7088, 7089, 7090, 7091, 7092, 7093, 7094, 7095, 7096, 7097, 7098],
    "What is (are) Leukemia": [7813, 7819, 7820, 7821, 7822, 7823, 7829, 7830, 7831],
    "What is (are) Breast Cancer": [518, 548, 658, 687, 7914, 7921, 7922, 7923, 7926, 7930, 7936],
    "What is (are) High Blood Cholesterol": [7962, 7965, 7966, 7967, 7968, 7969, 7970, 7971, 7972, 7973, 7975, 7976, 7977, 7978, 7979, 7980, 7983, 7984],
    "Who is at risk for Breast Cancer": [520, 549, 659, 688, 7915, 7916, 7924, 7927, 7931],
    "What is (are) Prostate Cancer": [53, 81, 96, 7653, 7659, 7660, 7662, 7674],
    "What is (are) Medicare and Continuing Care": [7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7871, 7872, 7873]
}

class CargadorCorpus:
    """Gestiona la carga y el acceso al corpus de documentos desde un archivo CSV."""

    def __init__(self):
        """Inicializa el cargador de corpus."""
        self.dfCorpus = None             # DataFrame principal del corpus
        self.indiceCorpus = None         # Mapeo de ID de documento a índice de fila (aunque es 1:1)
        self.numDocumentos = 0           # Número total de documentos
        self.mapeoQrels = {}            # {'pregunta': [id1, id2, ...]}
        self.listaQrels = []            # Lista de preguntas clave para la UI

    def cargarCorpus(self, nombreArchivoCsv: str = "corpus.csv", rutaRaiz: Path = None) -> bool:
        """
        Carga el corpus desde un único archivo CSV.

        Asume que el archivo CSV ya contiene las columnas "Question", "Answer" y "Topic".

        Args:
            nombreArchivoCsv: Nombre del archivo CSV que contiene el corpus unido.
                              Por defecto es "corpus_total.csv".
            rutaRaiz: Directorio raíz del proyecto. Si es None, utiliza la ruta relativa
                      para encontrar la carpeta 'docs'.

        Returns:
            bool: True si el corpus se carga con éxito, False en caso contrario.
        """
        try:
            if rutaRaiz is None:
                # Si se ejecuta desde controllers/corpus_loader.py, parents[1] es la raíz del proyecto
                rutaRaiz = Path(__file__).resolve().parents[1]
            else:
                rutaRaiz = Path(rutaRaiz)

            rutaDocs = rutaRaiz / "docs"
            rutaArchivo = rutaDocs / nombreArchivoCsv

            if not rutaArchivo.exists():
                logger.error(f"Archivo CSV no encontrado en: {rutaArchivo}")
                # Intenta buscar en la rutaRaiz directamente si no está en 'docs'
                rutaAlternativa = rutaRaiz / nombreArchivoCsv
                if rutaAlternativa.exists():
                    rutaArchivo = rutaAlternativa
                    logger.warning(f"Usando ruta alternativa: {rutaArchivo}")
                else:
                    logger.error(f"Archivo CSV no encontrado en la ruta raíz alternativa: {rutaAlternativa}")
                    return False
                
            # Cargar el DataFrame
            self.dfCorpus = pd.read_csv(rutaArchivo)
            
            # Restablecer el índice para asegurar que los doc_id coincidan con los índices de fila
            self.dfCorpus.reset_index(drop=True, inplace=True)
            
            # Crear índice de mapeo (doc_id -> row_index). Aquí es 1:1 (i:i)
            self.indiceCorpus = {i: i for i in range(len(self.dfCorpus))}
            self.numDocumentos = len(self.dfCorpus)
            
            # 4. Generar el mapeo de Qrels (Pregunta -> Lista de IDs de Documentos)
            self._generarMapeoQrels()
            
            logger.info(
                f"Corpus cargado con éxito desde '{nombreArchivoCsv}': {self.numDocumentos} documentos totales"
            )
            logger.debug(f"Columnas del Corpus: {list(self.dfCorpus.columns)}")
            
            return True

        except Exception as e:
            logger.error(f"Error al cargar el corpus desde {nombreArchivoCsv}: {str(e)}")
            self.dfCorpus = None # Asegurar que el estado sea limpio si hay error
            return False

    def obtenerDocumento(self, idDocumento: int) -> dict | None:
        """
        Recupera un documento por ID.

        Args:
            idDocumento: Índice del documento (entero).

        Returns:
            dict: Datos del documento como diccionario, o None si no se encuentra.
        """
        try:
            if self.dfCorpus is None:
                logger.warning("Corpus no cargado")
                return None

            if idDocumento not in self.indiceCorpus:
                logger.warning(f"ID de Documento {idDocumento} no encontrado en el corpus")
                return None

            indiceFila = self.indiceCorpus[idDocumento]
            fila = self.dfCorpus.iloc[indiceFila]
            
            return fila.to_dict()

        except Exception as e:
            logger.error(f"Error al recuperar el documento {idDocumento}: {str(e)}")
            return None

    def obtenerTodoElCorpus(self) -> pd.DataFrame | None:
        """
        Obtiene todos los documentos del corpus como un DataFrame.

        Returns:
            pd.DataFrame: DataFrame del Corpus o None si no está cargado.
        """
        return self.dfCorpus

    def buscarEnCorpus(self, idDocumentos: list[int], limite: int | None = None) -> list[dict]:
        """
        Recupera múltiples documentos por IDs.

        Args:
            idDocumentos: Lista de IDs de documentos.
            limite: Número máximo de documentos a devolver (equivalente al 'k' de ranking).

        Returns:
            list: Lista de diccionarios de documentos.
        """
        if limite is not None:
            idDocumentos = idDocumentos[:limite]

        documentos = []
        for idDoc in idDocumentos:
            documento = self.obtenerDocumento(idDoc)
            if documento:
                documentos.append(documento)

        return documentos

    def obtenerVistaPreviaDocumento(self, idDocumento: int, maxCaracteres: int = 200) -> str | None:
        """
        Obtiene una vista previa del contenido del documento.

        Args:
            idDocumento: ID del documento.
            maxCaracteres: Número máximo de caracteres a devolver.

        Returns:
            str: Texto de vista previa o "Contenido no disponible" si no se encuentra.
        """
        documento = self.obtenerDocumento(idDocumento)
        if not documento:
            return "Documento no encontrado"

        # Priorizar la columna 'Answer' para la vista previa
        textoVistaPrevia = None
        for campo in ["Answer", "answer", "Question", "question"]:
            if campo in documento and documento[campo]:
                textoVistaPrevia = str(documento[campo])
                break

        if textoVistaPrevia:
            if len(textoVistaPrevia) > maxCaracteres:
                return textoVistaPrevia[:maxCaracteres] + "..."
            return textoVistaPrevia

        return "Contenido no disponible"

    def estaCargado(self) -> bool:
        """Verifica si el corpus está cargado."""
        return self.dfCorpus is not None and len(self.dfCorpus) > 0
    
    def _generarMapeoQrels(self):
        """
        Asigna el mapeo de preguntas relevantes (Qrels) precalculado.
        """
        # Asignar directamente el diccionario precalculado
        self.mapeoQrels = QRELS_PRECALCULADOS
        
        # Generar la lista de preguntas clave y ordenarla
        self.listaQrels = sorted(list(self.mapeoQrels.keys()))

        logger.info(f"Mapeo Qrels cargado rápidamente: {len(self.listaQrels)} preguntas clave.")


    def obtenerListaQrels(self) -> List[str]:
        """ Retorna la lista de preguntas clave (Qrels). """
        return self.listaQrels

    def obtenerQrels(self, pregunta: str) -> List[int]:
        """ Retorna los IDs de documentos relevantes para una pregunta Qrel. """
        return self.mapeoQrels.get(pregunta, [])


# Instancia global del corpus
_instanciaCorpusGlobal = None


def obtenerCorpus() -> CargadorCorpus:
    """Obtiene o crea la instancia global del corpus."""
    global _instanciaCorpusGlobal
    if _instanciaCorpusGlobal is None:
        _instanciaCorpusGlobal = CargadorCorpus()
    return _instanciaCorpusGlobal


def inicializarCorpus(nombreArchivo: str = "corpus.csv", rutaRaiz: Path = None) -> bool:
    """
    Inicializa la instancia global del corpus.

    Args:
        nombreArchivo: Nombre del archivo CSV a cargar.
        rutaRaiz: Ruta del directorio raíz.

    Returns:
        bool: True si se carga con éxito.
    """
    corpus = obtenerCorpus()
    return corpus.cargarCorpus(nombreArchivo, rutaRaiz)