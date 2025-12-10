# 🔍 Retrieval Information Browser

Navegador de información con búsqueda usando tres modelos de recuperación de información: Binary Model, TF-IDF y BM25.

## 📋 Requisitos Previos

- Python 3.8+
- pip (gestor de paquetes de Python)
- Un entorno virtual (Opcional)

## Creación del entorno virtual

Ejecuta la siguiente línea para crear un entorno virtual de python llamado "env":

```powershell
python -m venv env
```

## 🚀 Instalación y Setup

### 1. Activar el Entorno Virtual

```powershell
c:\Users\USER\Retrieval-Inf-Project\env\Scripts\activate.bat
```

### 2. Instalar las librerías necesarias

```powershell
pip install -r requirements.txt
```

### 3. Agregar los modelos en la carpeta models

Ingresa al link y ejecuta los campos correspondientes cargando los archivos del corpus.

Archivos del corpus: https://www.kaggle.com/datasets/gvaldenebro/cancer-q-and-a-dataset?resource=download

Archivo en Google Colab: https://colab.research.google.com/drive/14IF7LH41EUthTZ88qjoXSCnO3Ly5IsZV?usp=sharing

Este último archivo genera un .csv llamado "corpus" que utiliza el proyecto y debe ser cargado en la carpeta docs

### 4. Descargar Datos de NLTK (Primera vez)

Ejecuta esto una sola vez para descargar los datos necesarios de NLTK:

```powershell
python setup_nltk.py
```

Este script descarga:

- `punkt_tab` - Tokenizador de palabras y oraciones
- `stopwords` - Palabras comunes en inglés

### 5. Ejecutar la Aplicación

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

- **TF-IDF / BM25:** `Doc {id} — score: {valor}` (documentos con scores)
- **Binary:** `Doc {id}` (documentos que coinciden)

## 📚 Corpus de Documentos

La aplicación carga automáticamente los documentos Q&A desde estos archivos CSV (en orden de concatenación):

1. **CancerQA.csv** - Q&A sobre cáncer
2. **Genetic_and_Rare_DiseasesQA.csv** - Q&A sobre enfermedades genéticas y raras
3. **Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv** - Q&A sobre diabetes y enfermedades digestivas
4. **SeniorHealthQA.csv** - Q&A sobre salud de adultos mayores

Cada documento contiene: Pregunta, Respuesta y Tópico.

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
│   ├── corpus.csv                # Preguntas y respuestas sobre todos los documentos
└── env/                          # Entorno virtual
```

## 📦 Dependencias

- `pandas` (usado para cargar y concatenar CSVs)
- `nltk`, `numpy`, `textual`, y librerías ya presentes en el entorno virtual

---

**Última actualización:** Diciembre 2025

```

```
