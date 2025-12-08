from typing import List, Tuple
from pathlib import Path
from .loadmodel import cargarModelo
from .corpus_loader import get_corpus
import logging

# Configurar logging
log_file = Path(__file__).resolve().parents[1] / "debug.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBrowser:
    """Simple bridge to load a pickled model and run searches.

    The class is intentionally minimal: it attempts to call the model's
    `buscar` method. It supports models that return either a list of
    (doc_id, score) tuples (TF-IDF / BM25) or an array/list of matching
    document indices (binary model).
    """

    def __init__(self) -> None:
        self.model = None
        self.model_path = None

        # Resolve project root (two levels up from this file: controllers/ -> project root)
        self.root = Path(__file__).resolve().parents[1]

    def load(self, path: str) -> Tuple[bool, str]:
        """Carga un archivo pickle usando cargarModelo.
        
        Retorna (éxito, mensaje).
        """
        # Convertir a ruta absoluta
        abs_path = Path(path).resolve()
        
        # Verificar que el archivo existe
        if not abs_path.exists():
            return False, f"El archivo no existe: {abs_path}"
        
        modelo = cargarModelo(str(abs_path))
        if modelo is None:
            return False, f"No se pudo cargar el modelo desde: {abs_path}"

        self.model = modelo
        self.model_path = str(abs_path)
        return True, f"Modelo cargado: {type(modelo).__name__}"

    def has_model(self) -> bool:
        return self.model is not None

    def list_models(self, models_dir: str = "models") -> List[Path]:
        """Return a list of Path objects for .pkl files in the project's models directory.

        The `models_dir` is relative to the project root by default.
        """
        folder = (self.root / models_dir).resolve()
        if not folder.exists() or not folder.is_dir():
            return []

        pkl_files = sorted(folder.glob("*.pkl"))
        return pkl_files

    def get_model_path(self, model_type: str, models_dir: str = "models") -> str:
        """Retorna la ruta del modelo según el tipo (binary, tfidf, bm25).
        
        Retorna la ruta absoluta como string, o string vacío si no lo encuentra.
        """
        folder = (self.root / models_dir).resolve()
        if not folder.exists() or not folder.is_dir():
            return ""

        # Mapeo directo de nombres de archivos (exactos)
        filename_map = {
            'binary': 'modeloBinario.pkl',
            'tfidf': 'modeloTfIdf.pkl',
            'bm25': 'modeloBM25.pkl'
        }

        model_type_lower = model_type.lower()
        filename = filename_map.get(model_type_lower)
        
        if filename:
            full_path = folder / filename
            if full_path.exists():
                return str(full_path)
        
        return ""

    def search(self, query: str, k: int = 5) -> List[str]:
        """Ejecuta una búsqueda contra el modelo cargado y retorna strings formateados.
        
        La lista retornada contiene líneas legibles para mostrar en la UI.
        """
        logger.debug(f"Iniciando búsqueda con query: '{query}'")
        
        if not self.model:
            logger.error("No hay modelo cargado")
            return []

        model_name = type(self.model).__name__
        logger.debug(f"Modelo en uso: {model_name}")
        
        # Intentar llamar a buscar con diferentes firmas de manera más inteligente
        result = None
        
        try:
            # ModeloBinario solo acepta la consulta
            if model_name == 'ModeloBinario':
                logger.debug(f"Llamando a ModeloBinario.buscar('{query}')")
                result = self.model.buscar(query)
            else:
                # ModeloVectorialTfIdf y ModeloBM25 aceptan k
                logger.debug(f"Llamando a {model_name}.buscar('{query}', k={k})")
                result = self.model.buscar(query, k)
                
            logger.debug(f"Resultado obtenido: tipo={type(result)}, len={len(result) if hasattr(result, '__len__') else 'N/A'}")
            
        except TypeError as e:
            logger.error(f"TypeError en primera llamada: {e}")
            # Fallback: intentar sin k
            try:
                logger.debug(f"Reintentando sin parámetro k")
                result = self.model.buscar(query)
                logger.debug(f"Resultado obtenido (reintento): tipo={type(result)}")
            except Exception as e2:
                logger.error(f"Error en reintento: {e2}", exc_info=True)
                return []
        except Exception as e:
            logger.error(f"Error general: {e}", exc_info=True)
            return []

        if result is None:
            logger.debug("Resultado es None")
            return []

        # Convertir numpy arrays a listas
        try:
            import numpy as np
            if isinstance(result, np.ndarray):
                logger.debug(f"Convirtiendo numpy array a lista")
                result = result.tolist()
        except Exception as e:
            logger.debug(f"No se pudo convertir numpy array: {e}")

        logger.debug(f"Tipo después de conversión: {type(result)}, len={len(result) if hasattr(result, '__len__') else 'N/A'}")
        formatted = []

        # Prepare corpus-based filtering: only keep docs whose 'Answer' contains all query tokens
        corpus = get_corpus()
        tokens = [t.strip().lower() for t in query.split() if t.strip()]

        def _answer_contains(doc_id: int) -> bool:
            try:
                doc = corpus.get_document(int(doc_id))
                if not doc:
                    return False
                # Try common Answer field names
                answer_text = None
                for key in ("Answer", "answer"):
                    if key in doc and isinstance(doc[key], str):
                        answer_text = doc[key].lower()
                        break
                if not answer_text:
                    return False
                # Require all tokens to be present in the answer (case-insensitive)
                for tok in tokens:
                    if tok not in answer_text:
                        return False
                return True
            except Exception:
                return False

        # Estilo TF-IDF / BM25: lista de tuplas (doc_id, score)
        if isinstance(result, list) and len(result) > 0:
            # Verificar si es lista de tuplas (TF-IDF/BM25)
            if isinstance(result[0], (tuple, list)) and len(result[0]) >= 2:
                try:
                    # Intentar acceder como tupla
                    _ = result[0][0], result[0][1]
                    logger.debug("Formato detectado: TF-IDF/BM25 (tuplas con scores)")
                    # Filter by Answer content
                    filtered = []
                    for item in result:
                        try:
                            docid = int(item[0])
                        except Exception:
                            continue
                        if tokens:
                            if _answer_contains(docid):
                                filtered.append(item)
                        else:
                            # No tokens (empty query tokens?) keep all
                            filtered.append(item)

                    if not filtered:
                        logger.debug("No TF-IDF/BM25 results pass Answer-column filtering")
                        return []

                    for item in filtered:
                        try:
                            docid = item[0]
                            score = item[1]
                            formatted.append(f"Doc {docid} — score: {float(score):.4f}")
                        except Exception:
                            formatted.append(str(item))
                    logger.debug(f"{len(formatted)} resultados formateados (TF-IDF/BM25)")
                    return formatted
                except (TypeError, IndexError):
                    pass

        # Estilo Binary: lista de índices (números)
        logger.debug("Formateando como índices (Binary)")
        try:
            filtered_ids = []
            for i in result:
                try:
                    docid = int(i)
                except Exception:
                    continue
                if tokens:
                    if _answer_contains(docid):
                        filtered_ids.append(docid)
                else:
                    filtered_ids.append(docid)

            if not filtered_ids:
                logger.debug("No Binary results pass Answer-column filtering")
                return []

            for docid in filtered_ids:
                formatted.append(f"Doc {int(docid)}")
            logger.debug(f"{len(formatted)} resultados formateados (Binary)")
            return formatted
        except Exception as e:
            logger.error(f"Error al formatear como índices: {e}", exc_info=True)
            logger.debug(f"Tipo final de result: {type(result)}")
            return [str(result)]
