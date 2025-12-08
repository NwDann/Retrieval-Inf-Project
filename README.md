# 🔍 Retrieval Information Browser

Navegador de información con búsqueda usando tres modelos de recuperación de información: Binary Model, TF-IDF y BM25.

## 📋 Requisitos Previos

- Python 3.8+
- pip (gestor de paquetes de Python)

## 🚀 Instalación y Setup

### 1. Activar el Entorno Virtual

```powershell
c:\Users\USER\Retrieval-Inf-Project\env\Scripts\Activate.ps1
```

### 2. Descargar Datos de NLTK (Primera vez)

Ejecuta esto una sola vez para descargar los datos necesarios de NLTK:

```powershell
python setup_nltk.py
```

Este script descarga:
- `punkt_tab` - Tokenizador de palabras y oraciones
- `stopwords` - Palabras comunes en inglés

### 3. Ejecutar la Aplicación

```powershell
python main.py
```

## 🎯 Cómo Usar

1. **Selecciona un modelo de búsqueda:**
   - **Binary Model** - Búsqueda booleana (AND): devuelve documentos donde TODOS los términos están presentes
   - **TF-IDF Model** - Modelo vectorial: devuelve documentos rankeados por similitud
   - **BM25 Model** - Modelo probabilístico: devuelve documentos rankeados por probabilidad

2. **Ingresa una consulta** en el campo "Buscar..." (ej: "cancer", "diabetes")

3. **Haz clic en "Buscar"** para ver los resultados

4. **Haz clic en un resultado** para ver el documento completo con todos sus campos (pregunta, respuesta, tópico, etc.)

**Formato de resultados:**
- **TF-IDF / BM25:** `Doc {id} — score: {valor}`  (documentos con scores)
- **Binary:** `Doc {id}` (documentos que coinciden)

## 📚 Corpus de Documentos

La aplicación carga automáticamente los documentos Q&A desde estos archivos CSV (en orden de concatenación):

1. **CancerQA.csv** - Q&A sobre cáncer
2. **Genetic_and_Rare_DiseasesQA.csv** - Q&A sobre enfermedades genéticas y raras
3. **Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv** - Q&A sobre diabetes y enfermedades digestivas
4. **SeniorHealthQA.csv** - Q&A sobre salud de adultos mayores

Cada documento contiene: Pregunta, Respuesta, Tópico y Split.

Los documentos se indexan automáticamente al iniciar la aplicación.

## 📁 Estructura del Proyecto

```
Retrieval-Inf-Project/
├── main.py                       # Aplicación principal (Textual UI)
├── setup_nltk.py                 # Script para descargar datos NLTK
├── test_models.py                # Script de prueba de modelos (sin UI)
├── test_corpus.py                # Script de prueba del corpus
├── test_integration.py           # Test de integración completo
├── controllers/
│   ├── loadmodel.py              # Cargador de modelos pickle
│   ├── browser_integration.py    # Lógica de búsqueda
│   └── corpus_loader.py          # Cargador de corpus desde CSVs
├── classes/
│   ├── binarymodel.py            # Modelo Binary
│   ├── tfidfmodel.py             # Modelo TF-IDF
│   └── bm25model.py              # Modelo BM25
├── models/
│   ├── modeloBinario.pkl         # Modelo Binary entrenado
│   ├── modeloTfIdf.pkl           # Modelo TF-IDF entrenado
│   └── modeloBM25.pkl            # Modelo BM25 entrenado
├── docs/
│   ├── CancerQA.csv              # Preguntas y respuestas sobre cáncer
│   ├── Genetic_and_Rare_DiseasesQA.csv
│   ├── Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv
│   └── SeniorHealthQA.csv        # Preguntas y respuestas sobre salud de adultos mayores
└── env/                          # Entorno virtual
```

## 🧪 Testing (Opcional)

### Test de Modelos (sin UI)
```powershell
python test_models.py
```
Verifica que todos los modelos cargan correctamente y realiza búsquedas de prueba.

### Test de Corpus
```powershell
python test_corpus.py
```
Verifica que el corpus se carga correctamente desde los archivos CSV.

### Test de Integración Completo
```powershell
python test_integration.py
```
Prueba el flujo completo: carga de corpus → búsqueda con modelo → acceso a documentos.

## 🔧 Troubleshooting

### Error: `Resource punkt_tab not found`
- Solución: Ejecuta `python setup_nltk.py`

### Error: `No se encontró modelo`
```markdown
# 🔍 Retrieval Information Browser

Navegador de información con búsqueda usando tres modelos de recuperación de información: Binary Model, TF-IDF y BM25.

## 📋 Requisitos Previos

- Python 3.8+
- pip (gestor de paquetes de Python)

## 🚀 Instalación y Setup

### 1. Activar el Entorno Virtual

```powershell
c:\Users\USER\Retrieval-Inf-Project\env\Scripts\Activate.ps1
```

### 2. Descargar Datos de NLTK (Primera vez)

Ejecuta esto una sola vez para descargar los datos necesarios de NLTK:

```powershell
python setup_nltk.py
```

Este script descarga:
- `punkt_tab` - Tokenizador de palabras y oraciones
- `stopwords` - Palabras comunes en inglés

### 3. Ejecutar la Aplicación

```powershell
python main.py
```

## 🎯 Cómo Usar

1. **Selecciona un modelo de búsqueda:**
   - **Binary Model** - Búsqueda booleana (AND): devuelve documentos donde TODOS los términos están presentes
   - **TF-IDF Model** - Modelo vectorial: devuelve documentos rankeados por similitud
   - **BM25 Model** - Modelo probabilístico: devuelve documentos rankeados por probabilidad

2. **Ingresa una consulta** en el campo "Buscar..." (ej: "cancer", "diabetes")

3. **Haz clic en "Buscar"** para ver los resultados

4. **Haz clic en un resultado** para ver el documento completo con todos sus campos (pregunta, respuesta, tópico, etc.)

**Formato de resultados:**
- **TF-IDF / BM25:** `Doc {id} — score: {valor}`  (documentos con scores)
- **Binary:** `Doc {id}` (documentos que coinciden)

## 📚 Corpus de Documentos

La aplicación carga automáticamente los documentos Q&A desde estos archivos CSV (en orden de concatenación):

1. **CancerQA.csv** - Q&A sobre cáncer
2. **Genetic_and_Rare_DiseasesQA.csv** - Q&A sobre enfermedades genéticas y raras
3. **Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv** - Q&A sobre diabetes y enfermedades digestivas
4. **SeniorHealthQA.csv** - Q&A sobre salud de adultos mayores

Cada documento contiene: Pregunta, Respuesta, Tópico y Split.

Los documentos se indexan automáticamente al iniciar la aplicación.

## 📁 Estructura del Proyecto

```
Retrieval-Inf-Project/
├── main.py                       # Aplicación principal (Textual UI)
├── setup_nltk.py                 # Script para descargar datos NLTK
├── controllers/
│   ├── loadmodel.py              # Cargador de modelos pickle
│   ├── browser_integration.py    # Lógica de búsqueda
│   └── corpus_loader.py          # Cargador de corpus desde CSVs
├── classes/
│   ├── binarymodel.py            # Modelo Binary
│   ├── tfidfmodel.py             # Modelo TF-IDF
│   └── bm25model.py              # Modelo BM25
├── models/
│   ├── modeloBinario.pkl         # Modelo Binary entrenado
│   ├── modeloTfIdf.pkl           # Modelo TF-IDF entrenado
│   └── modeloBM25.pkl            # Modelo BM25 entrenado
├── docs/
│   ├── CancerQA.csv              # Preguntas y respuestas sobre cáncer
│   ├── Genetic_and_Rare_DiseasesQA.csv
│   ├── Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv
│   └── SeniorHealthQA.csv        # Preguntas y respuestas sobre salud de adultos mayores
└── env/                          # Entorno virtual (no recomendado en repo)
```

## 🧪 Testing (Opcional)

Los tests son útiles para desarrollo. Puedes conservarlos o eliminarlos según prefieras.

### Test de Modelos (sin UI)
```powershell
python test_models.py
```

### Test de Corpus
```powershell
python test_corpus.py
```

### Test de Integración Completo
```powershell
python test_integration.py
```

### Validación Rápida
```powershell
python validate.py
```

## 🔧 Troubleshooting

### Error: `Resource punkt_tab not found`
- Solución: Ejecuta `python setup_nltk.py`

### Error: `No se encontró modelo`
- Verifica que los archivos `.pkl` estén en la carpeta `models/`

### La búsqueda devuelve muchos resultados
- Esto es normal con modelos grandes
- Los resultados se muestran en orden de relevancia

## 📝 Archivos Importantes

| Archivo | Propósito | ¿Necesario? |
|---------|-----------|-----------|
| `main.py` | App principal | ✅ Sí |
| `setup_nltk.py` | Descargar datos NLTK | ✅ Sí (una sola vez) |
| `test_models.py` | Testing de modelos | ⚠️ Opcional |
| `test_corpus.py` | Testing de corpus | ⚠️ Opcional |
| `test_integration.py` | Testing integración completa | ⚠️ Opcional |
| `validate.py` | Validación rápida | ⚠️ Opcional |
| `debug.log` | Log de debugging | ❌ No (se genera automáticamente) |

## 📚 Documentación de Módulos

### `browser_integration.py`

Clase `ModelBrowser`:
- `load(path)` - Carga un modelo desde una ruta
- `get_model_path(type)` - Obtiene la ruta de un modelo por tipo
- `search(query, k)` - Ejecuta una búsqueda
- `has_model()` - Verifica si hay modelo cargado

### `loadmodel.py`

Función `cargarModelo(nombreArchivo)`:
- Carga modelos pickle con manejo especial de clases
- Retorna el modelo cargado o `None` si hay error

### `corpus_loader.py`

Clase `CorpusLoader`:
- `load_corpus(root_path)` - Carga y concatena CSVs en orden específico
- `get_document(doc_id)` - Recupera un documento por ID
- `get_all_documents()` - Obtiene el DataFrame completo
- `get_document_preview(doc_id, max_chars)` - Obtiene una vista previa del documento
- `is_loaded()` - Verifica si el corpus está cargado

Funciones globales:
- `initialize_corpus(root_path)` - Inicializa la instancia global del corpus
- `get_corpus()` - Obtiene la instancia global del corpus

## 📋 Resumen de Cambios Recientes

Se implementaron las siguientes mejoras principales:

- Nuevo módulo `controllers/corpus_loader.py` para gestionar el corpus desde CSVs.
- Integración en `main.py` para cargar el corpus al iniciar y mostrar documentos completos al hacer click.
- Scripts de prueba (`test_corpus.py`, `test_integration.py`, `test_models.py`) y un pequeño `validate.py` para validación rápida.

## 📦 Dependencias

- `pandas` (usado para cargar y concatenar CSVs)
- `nltk`, `numpy`, `textual`, y librerías ya presentes en el entorno virtual

---

**Última actualización:** Diciembre 2025

``` 
