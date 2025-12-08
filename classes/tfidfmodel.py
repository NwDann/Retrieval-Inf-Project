import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class ModeloVectorialTfIdf:
    """
    Implementación del Modelo Vectorial utilizando la ponderación TF-IDF.
    """
    def __init__(self):
        self.vocabulario = {}        # Término a ID (índice de columna)
        self.vectorIdf = None        # Vector de NumPy con los pesos IDF
        self.matrizTfIdf = None       # Matriz de NumPy (Documentos x Términos)
        self.listaStopwords = set(stopwords.words("english"))
        self.listaDocumentos = []    # Lista de IDs/Índices de documentos
        self.numDocumentos = 0       # Total de documentos en el corpus

    def preProcesar(self, texto):
        """ Tokenización y eliminación de stopwords (reutilizado). """
        textoMin = texto.lower()
        tokens = word_tokenize(textoMin)
        tokensFiltrados = [
            token for token in tokens
            if token.isalpha() and token not in self.listaStopwords
        ]
        return tokensFiltrados

    # --- Ponderación del Modelo ---

    def calcularTf(self, docTokens):
        """ Calcula la Frecuencia de Término (TF) para un documento. """
        frecuencias = {}
        for token in docTokens:
            frecuencias[token] = frecuencias.get(token, 0) + 1

        # normalización (por longitud del documento)
        # longitud = len(docTokens)
        # tfVector = np.array([frecuencias.get(t, 0) / longitud for t in self.vocabulario.keys()])
        return frecuencias

    def calcularIdf(self, matrizTf):
        """ Calcula la Frecuencia Inversa de Documento (IDF) para todos los términos. """
        # Frecuencia de documento (df): cuántos documentos contienen el término
        # Sumamos las ocurrencias (0s y 1s) a lo largo de los documentos (eje 0)
        documentosConTermino = np.sum(matrizTf > 0, axis=0)

        # division por cero
        # log(N / df_t) + 1
        idfVector = np.log((self.numDocumentos + 1) / (documentosConTermino + 1)) + 1

        return idfVector

    def normalizarMatriz(self, matriz):
        """ Normaliza los vectores de la matriz a longitud unitaria (norma L2). """
        # Calcular la norma euclidiana (L2-norm) de cada fila (vector de documento)
        normas = np.linalg.norm(matriz, axis=1)
        # Para evitar división por cero, solo dividimos donde la norma es > 0
        matrizNormalizada = np.divide(
            matriz,
            normas[:, np.newaxis],
            out=np.zeros_like(matriz, dtype=float), # Si la norma es 0, deja el vector como 0
            where=normas[:, np.newaxis]!=0
        )
        return matrizNormalizada

    def ajustarCorpus(self, serieDocumentos):
        """
        Crea el vocabulario, la matriz de frecuencia y calcula la matriz TF-IDF final.
        """

        documentosTokenizados = []
        for docId, texto in enumerate(serieDocumentos):
            tokens = self.preProcesar(texto)
            documentosTokenizados.append(tokens)
            self.listaDocumentos.append(docId)
            for token in tokens:
                if token not in self.vocabulario:
                    self.vocabulario[token] = len(self.vocabulario)

        self.numDocumentos = len(self.listaDocumentos)
        numTerminos = len(self.vocabulario)

        # matriz de Frecuencia de Término (Count Matrix)
        matrizFrecuencia = np.zeros((self.numDocumentos, numTerminos), dtype=np.int32)
        for docIndex, tokens in enumerate(documentosTokenizados):
            frecuencias = self.calcularTf(tokens)
            for token, freq in frecuencias.items():
                if token in self.vocabulario:
                    terminoIndex = self.vocabulario[token]
                    matrizFrecuencia[docIndex, terminoIndex] = freq

        # calcular IDF
        self.vectorIdf = self.calcularIdf(matrizFrecuencia)

        # calcular Matriz TF-IDF (Term Frequency * Inverse Document Frequency)
        # multiplicación elemento a elemento de la matriz TF por el vector IDF (broadcasting)
        matrizTfIdfCruda = matrizFrecuencia * self.vectorIdf

        # normalizar la Matriz TF-IDF
        self.matrizTfIdf = self.normalizarMatriz(matrizTfIdfCruda)

        print(f"Ajuste completado. Documentos: {self.numDocumentos}, Términos: {numTerminos}")
        print("Muestra de la Matriz TF-IDF (Normalizada):")
        print(self.matrizTfIdf)

    # --- Búsqueda (Search) del Modelo ---

    def buscar(self, consulta, k=3):
        """
        Calcula la similitud de la consulta con todos los documentos (Similitud del Coseno)
        y devuelve los 'k' documentos más relevantes.
        """
        print(f"\nBuscando (TF-IDF): '{consulta}'")
        tokensConsulta = self.preProcesar(consulta)

        # 1. Convertir la consulta a un vector TF-IDF
        vectorConsulta = np.zeros(len(self.vocabulario), dtype=float)

        # Calcular TF de la consulta
        frecuenciasConsulta = self.calcularTf(tokensConsulta)

        for token, freq in frecuenciasConsulta.items():
            if token in self.vocabulario:
                terminoIndex = self.vocabulario[token]
                # Ponderación TF-IDF: TF de la consulta * IDF del corpus
                vectorConsulta[terminoIndex] = freq * self.vectorIdf[terminoIndex]

        # 2. Normalizar el vector de consulta
        # La norma del vector de consulta
        normaConsulta = np.linalg.norm(vectorConsulta)
        if normaConsulta > 0:
            vectorConsultaNormalizado = vectorConsulta / normaConsulta
        else:
            return []

        # 3. Calcular Similitud del Coseno
        # Similitud del Coseno = A . B / (||A|| * ||B||)
        # Como ambos (matrizTfIdf y vectorConsultaNormalizado) ya están normalizados (norma 1),
        # la Similitud del Coseno es simplemente el producto punto:
        # Cos(theta) = MatrizTfIdf . VectorConsultaNormalizado_transpuesto

        # Producto punto entre la matriz (D x T) y el vector (T)
        similitudes = self.matrizTfIdf @ vectorConsultaNormalizado.T

        # 4. Obtener los índices de los documentos ordenados por similitud (descendente)
        # np.argsort devuelve los índices que ordenarían el array
        indicesOrdenados = np.argsort(similitudes)[::-1]

        # Obtener las puntuaciones de los documentos relevantes (top K)
        topKIndices = indicesOrdenados[:k]
        topKScores = similitudes[topKIndices]

        resultados = [(self.listaDocumentos[i], topKScores[idx]) for idx, i in enumerate(topKIndices)]

        print(f"Top {k} resultados encontrados (ID, Similitud del Coseno):")
        return resultados