import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Definición de la clase BM25

class ModeloBM25:
    """
    Implementación del Modelo de Ranking BM25 utilizando solo NumPy.
    """

    def __init__(self, k1=1.2, b=0.75, idioma='spanish'):
        self.k1 = k1                       # Parámetro de ajuste de saturación de TF
        self.b = b                         # Parámetro de ajuste de normalización por longitud
        self.vocabulario = {}              # Término a ID (índice de columna)
        self.listaStopwords = set(stopwords.words(idioma))
        self.listaDocumentos = []          # Lista de IDs/Índices de documentos
        self.matrizFrecuencia = None       # Matriz de NumPy (Documentos x Términos)
        self.vectorLongitudDocumento = None# Vector con la longitud de cada documento |D|
        self.longitudPromedio = 0.0        # Longitud promedio de los documentos avgdl
        self.vectorIdf = None              # Vector de NumPy con los pesos IDF de BM25

    def preProcesar(self, texto):
        """ Tokenización y eliminación de stopwords. """
        textoMin = texto.lower()
        tokens = word_tokenize(textoMin)
        tokensFiltrados = [
            token for token in tokens
            if token.isalpha() and token not in self.listaStopwords
        ]
        return tokensFiltrados

    # --- Ajuste (Fit) del Modelo ---

    def ajustarCorpus(self, serieDocumentos):
        """
        Crea el vocabulario, calcula las longitudes de documento, IDF,
        y la matriz de frecuencia necesaria para la puntuación.
        """
        print("\nIniciando ajuste del Modelo BM25...")

        documentosTokenizados = []
        longitudes = []

        # 1. Tokenizar, generar vocabulario y calcular longitudes
        for docId, texto in enumerate(serieDocumentos):
            tokens = self.preProcesar(texto)
            documentosTokenizados.append(tokens)
            longitudes.append(len(tokens))
            self.listaDocumentos.append(docId)
            for token in tokens:
                if token not in self.vocabulario:
                    self.vocabulario[token] = len(self.vocabulario)

        self.vectorLongitudDocumento = np.array(longitudes, dtype=float)
        self.numDocumentos = len(self.listaDocumentos)
        self.longitudPromedio = np.mean(self.vectorLongitudDocumento)
        numTerminos = len(self.vocabulario)

        # 2. Crear Matriz de Frecuencia de Término (Count Matrix)
        self.matrizFrecuencia = np.zeros((self.numDocumentos, numTerminos), dtype=np.int32)
        for docIndex, tokens in enumerate(documentosTokenizados):
            for token in tokens:
                if token in self.vocabulario:
                    terminoIndex = self.vocabulario[token]
                    self.matrizFrecuencia[docIndex, terminoIndex] += 1

        # 3. Calcular IDF (Específico de BM25)
        # BM25 IDF: log( (N - df_t + 0.5) / (df_t + 0.5) )
        documentosConTermino = np.sum(self.matrizFrecuencia > 0, axis=0) # df_t
        N = self.numDocumentos

        # Uso de NumPy para aplicar la fórmula a todos los términos
        self.vectorIdf = np.log((N - documentosConTermino + 0.5) / (documentosConTermino + 0.5))

        print(f"Ajuste completado. Documentos: {self.numDocumentos}, Términos: {numTerminos}")
        print(f"Longitud Promedio (avgdl): {self.longitudPromedio:.2f}")


    # --- Búsqueda (Search) del Modelo ---

    def buscar(self, consulta, k=3):
        """
        Calcula las puntuaciones BM25 para la consulta y ranquea los documentos.
        """
        print(f"\nBuscando (BM25): '{consulta}'")
        tokensConsulta = self.preProcesar(consulta)
        puntuaciones = np.zeros(self.numDocumentos, dtype=float)

        # Normalización por longitud (B): k1 * (1 - b + b * (|D| / avgdl))
        normalizacionDoc = self.k1 * (
            (1 - self.b) + self.b * (self.vectorLongitudDocumento / self.longitudPromedio)
        )

        for token in tokensConsulta:
            if token in self.vocabulario:
                terminoIndex = self.vocabulario[token]
                # Obtener la columna de IDF y la columna de frecuencia (tf)
                idfTermino = self.vectorIdf[terminoIndex]
                frecuenciasTermino = self.matrizFrecuencia[:, terminoIndex] # f(t_i, D)

                # Expresión del numerador de la fórmula: f(t_i, D) * (k1 + 1)
                numerador = frecuenciasTermino * (self.k1 + 1)

                # Expresión del denominador: f(t_i, D) + Normalización por longitud
                denominador = frecuenciasTermino + normalizacionDoc

                # Acumular puntuaciones (IDF * Factor de saturación/normalización)
                # Solo sumamos para los términos que tienen un IDF positivo (términos relevantes)
                if idfTermino > 0:
                    puntuaciones += idfTermino * (numerador / denominador)

        # 1. Obtener los índices de los documentos ordenados por puntuación (descendente)
        indicesOrdenados = np.argsort(puntuaciones)[::-1]

        # 2. Seleccionar los top K documentos con puntuaciones > 0
        topKIndices = [i for i in indicesOrdenados if puntuaciones[i] > 0][:k]
        topKScores = puntuaciones[topKIndices]

        if len(topKIndices) == 0:
            print("No se encontraron documentos relevantes.")
            return []

        resultados = [(self.listaDocumentos[i], topKScores[idx]) for idx, i in enumerate(topKIndices)]

        print(f"Top {k} resultados encontrados (ID, Puntuación BM25):")
        return resultados