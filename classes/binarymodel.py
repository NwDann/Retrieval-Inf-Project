import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class ModeloBinario:
    """
    Utiliza una matriz de ocurrencia término-documento.
    """

    def __init__(self):
        self.vocabulario = {} # Diccionario de término a ID
        self.matrizOcurrencia = None # Matriz de NumPy (Documentos x Términos)
        self.listaDocumentos = [] # Lista de IDs/Índices de documentos
        self.listaStopwords = set(stopwords.words('english'))

    def preProcesar(self, texto):
        """ Tokenización y eliminación de stopwords para un texto. """
        # Convertir a minúsculas
        textoMin = texto.lower()
        # Tokenizar (NLTK es permitido)
        tokens = word_tokenize(textoMin)
        # Filtrar stopwords y tokens no alfabéticos
        tokensFiltrados = [
            token for token in tokens
            if token.isalpha() and token not in self.listaStopwords
        ]
        return tokensFiltrados

    def ajustarCorpus(self, serieDocumentos):
        """
        'Ajusta' el modelo al corpus, creando el vocabulario y la matriz.
        serieDocumentos debe ser la columna 'Answer' del DataFrame.
        """
        print("\nIniciando ajuste del Modelo Binario...")
        documentosTokenizados = []
        documentoID = 0

        # 1. Generar tokens y vocabulario
        for texto in serieDocumentos:
            tokens = self.preProcesar(texto)
            documentosTokenizados.append(tokens)
            self.listaDocumentos.append(documentoID) # Usamos el índice de la serie como ID
            documentoID += 1

            for token in tokens:
                if token not in self.vocabulario:
                    # Asignar un ID único a cada término
                    self.vocabulario[token] = len(self.vocabulario)

        # 2. Crear Matriz de Ocurrencia (Documentos x Términos)
        numDocs = len(documentosTokenizados)
        numTerminos = len(self.vocabulario)

        # Inicializar la matriz con ceros
        self.matrizOcurrencia = np.zeros((numDocs, numTerminos), dtype=np.int8)

        # Llenar la matriz con 1s para indicar presencia
        for docIndex, tokens in enumerate(documentosTokenizados):
            for token in tokens:
                # Obtener el ID del término
                terminoIndex = self.vocabulario[token]
                # Marcar presencia (1)
                self.matrizOcurrencia[docIndex, terminoIndex] = 1

        print(f"Ajuste completado. Documentos: {numDocs}, Términos: {numTerminos}")
        print("Matriz de Ocurrencia (Documentos x Términos):")
        print(self.matrizOcurrencia)

    def buscar(self, consulta, k=3):
        """
        Realiza una búsqueda simple (AND) y devuelve los primeros k resultados.
        """
        print(f"\nBuscando: '{consulta}' con límite k={k}")
        tokensConsulta = self.preProcesar(consulta)

        # Máscara binaria para la relevancia: todos los documentos inicialmente relevantes
        relevanciaBooleana = np.ones(len(self.listaDocumentos), dtype=bool)

        terminosNoEncontrados = []

        # 1. Aplicar la lógica AND para cada término de la consulta
        for token in tokensConsulta:
            if token in self.vocabulario:
                terminoIndex = self.vocabulario[token]
                vectorTermino = self.matrizOcurrencia[:, terminoIndex]

                # Operación AND
                relevanciaBooleana = relevanciaBooleana & (vectorTermino == 1)
            else:
                terminosNoEncontrados.append(token)
                relevanciaBooleana[:] = False
                break

        # 2. Obtener los índices de los documentos relevantes
        indicesRelevantes = np.where(relevanciaBooleana)[0]

        # 3. Aplicar el límite k
        # Como es un modelo binario, no hay ranking, simplemente tomamos los primeros 'k'
        indicesLimitados = indicesRelevantes[:k]

        if len(indicesRelevantes) == 0:
            print("No se encontraron documentos relevantes.")
            return []

        print(f"Documentos relevantes encontrados (total): {len(indicesRelevantes)}")
        print(f"Documentos retornados (top k={k}): {len(indicesLimitados)}")
        print(f"Top {k} resultados encontrados (ID):")
        return indicesLimitados