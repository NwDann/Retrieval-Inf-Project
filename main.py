from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer, Button, Input, ListItem, ListView, Label, Static, Select
from textual.message import Message
from typing import List

# Importaciones de los controladores con sus nombres traducidos (si los archivos fueran renombrados)
# Usaremos los nombres de clase traducidos para mantener la coherencia.
from controllers.browser_integration import NavegadorModelos
from controllers.corpus_loader import inicializarCorpus, obtenerCorpus


class ResultadoBusqueda(ListItem):
    """Elemento clicable de un resultado de búsqueda."""
    
    class Seleccionado(Message):
        """Mensaje emitido cuando se selecciona este elemento."""
        def __init__(self, texto: str):
            # No pasar el remitente a Message.__init__ — Textual lo establecerá al publicar
            super().__init__()
            self.texto = texto

    def __init__(self, texto: str):
        super().__init__(Label(texto))
        self.texto = texto

    def on_click(self) -> None:
        """Maneja el evento de click y publica el mensaje Seleccionado."""
        # Publicar solo el texto; Textual adjuntará el remitente automáticamente
        self.post_message(self.Seleccionado(self.texto))


class Navegador(Static):
    """Contenedor estático para los controles de selección de modelo y búsqueda."""

    def compose(self) -> ComposeResult:
        """Define los widgets hijos de este componente."""
        # Selector de tipo de modelo
        yield Container(
            Label("Selecciona el tipo de modelo:"),
            Button("Modelo Binario", id="seleccionar_binario_boton"),
            Button("Modelo TF-IDF", id="seleccionar_tfidf_boton"),
            Button("Modelo BM25", id="seleccionar_bm25_boton"),
            Label("", id="etiqueta_modelo_seleccionado"),
            classes="selector_modelos" # <-- NUEVA CLASE PARA CSS
        )
        
        # Selector Qrels
        yield Container(
            Label("O selecciona una pregunta de Evaluación (Qrel):"),
            Select([], id="selector_qrel"),
            classes="qrel_container"
        )

        # Área de búsqueda
        # Definiremos esta columna en el siguiente punto para incluir 'k'
        yield Horizontal(
            Container( # Nuevo contenedor para agrupar k y Buscar
                Label("Límite K:"),
                Input(value="5", type="integer", id="entrada_k", classes="entrada_k"),
                classes="k_input_container"
            ),
            Input(placeholder="Buscar...", id="entrada_busqueda", classes="input_horizontal"),
            Button("Buscar", id="buscar_boton", classes="button_horizontal"),
            classes="controles_busqueda_horizontal"
        )
        
        # --- LISTA DE RESULTADOS (Ocupa todo el ancho) ---
        yield ListView(id="lista_resultados")


class Camaleon(App):
    """Aplicación principal de Textual para el proyecto de IR."""
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("q", "quit", "Salir"),
        ("s", "toggle_dark", "Alternar Modo Oscuro") # Un ejemplo de binding útil
    ]

    def compose(self) -> ComposeResult:
        """Define la disposición y los widgets principales de la aplicación."""
        yield Header(show_clock=True)
        yield Navegador()
        yield Footer()

    def on_mount(self) -> None:
        """Llamado una vez que los widgets están montados en el DOM."""
        # Crear el puente (bridge) del navegador de modelos
        self.navegadorModelos = NavegadorModelos()
        self.tipoModeloSeleccionado = None
        
        # Cargar corpus desde CSV (usando la función de corpus_loader.py)
        if inicializarCorpus():
             self.notify("✓ Corpus cargado con éxito", severity="information")
             
             # 2. Llenar el selector de Qrels
             corpus = obtenerCorpus()
             preguntasQrel = corpus.obtenerListaQrels()
             
             opcionesQrel = [(p, p) for p in preguntasQrel] # (Texto, Valor)
             
             selectorQrel: Select = self.query_one("#selector_qrel")
             # El primer elemento será una instrucción (None)
             selectorQrel.set_options([("Seleccionar Qrel", None)] + opcionesQrel)
             
        else:
             self.notify("⚠ Advertencia: El corpus no pudo ser cargado", severity="warning")
        
        # Verificar modelos disponibles
        self.refrescarListaModelos()

    def on_button_pressed(self, evento: Button.Pressed) -> None:
        """Maneja los clicks de los botones."""
        if evento.button.id == "buscar_boton":
            self.ejecutarBusqueda()
        elif evento.button.id == "seleccionar_binario_boton":
            self.seleccionarTipoModelo("binary")
        elif evento.button.id == "seleccionar_tfidf_boton":
            self.seleccionarTipoModelo("tfidf")
        elif evento.button.id == "seleccionar_bm25_boton":
            self.seleccionarTipoModelo("bm25")
    
    def on_select_changed(self, evento: Select.Changed) -> None:
        """Maneja cuando se selecciona una pregunta Qrel."""
        if evento.select.id == "selector_qrel" and evento.value is not None:
            # Copiar el valor seleccionado al campo de búsqueda
            entradaBusqueda: Input = self.query_one("#entrada_busqueda")
            entradaBusqueda.value = str(evento.value)
            self.notify(f"Qrel seleccionado: '{evento.value}'", severity="information")

    def seleccionarTipoModelo(self, tipoModelo: str) -> None:
        """Selecciona un tipo de modelo (binary, tfidf, bm25) y lo carga."""
        # Usar el método del navegador de modelos para obtener la ruta
        ruta = self.navegadorModelos.obtenerRutaModelo(tipoModelo)
        
        if not ruta:
            self.notify(f"✗ No se encontró modelo {tipoModelo}", severity="error")
            return

        exito, mensaje = self.navegadorModelos.cargar(ruta)
        
        etiqueta: Label = self.query_one("#etiqueta_modelo_seleccionado")
        listaResultados: ListView = self.query_one("#lista_resultados")
        listaResultados.clear()
        
        if exito:
            self.notify(f"✓ {mensaje}", severity="information")
            self.tipoModeloSeleccionado = tipoModelo
            etiqueta.update(f"✓ Modelo {tipoModelo.upper()} Cargado")
            listaResultados.append(ResultadoBusqueda(mensaje))
        else:
            self.notify(f"✗ {mensaje}", severity="error")
            etiqueta.update(f"✗ Error al cargar modelo")
            listaResultados.append(ResultadoBusqueda(f"Error: {mensaje}"))

    def refrescarListaModelos(self) -> None:
        """Verifica modelos disponibles en la carpeta models/."""
        entradas = self.navegadorModelos.listarModelos()
        if entradas:
            self.notify(f"✓ Se encontraron {len(entradas)} modelos disponibles", severity="information")
        else:
            self.notify("✗ No hay archivos .pkl en la carpeta models/", severity="warning")

    def ejecutarBusqueda(self):
        """Ejecuta la búsqueda usando el modelo seleccionado."""
        listaResultados: ListView = self.query_one("#lista_resultados")
        listaResultados.clear()

        entradaBusqueda: Input = self.query_one("#entrada_busqueda")
        consulta = (entradaBusqueda.value or "").strip()
        
        entradaK: Input = self.query_one("#entrada_k")
        
        try:
            # Convertir a entero, con un mínimo de 1
            k = max(1, int(entradaK.value))
        except ValueError:
            self.notify("K debe ser un número entero válido (mínimo 1)", severity="error")
            return

        if not consulta:
            msg = "Por favor ingresa una consulta de búsqueda"
            self.notify(msg, severity="warning")
            return

        if not self.navegadorModelos.tieneModelo():
            msg = "Por favor selecciona y carga un modelo primero"
            self.notify(msg, severity="warning")
            return

        # Ejecutar búsqueda (usando k=5 como límite por defecto)
        self.notify(f"Buscando '{consulta}' con K={k}...", severity="information")
        lineasFormateadas: List[str] = self.navegadorModelos.buscar(consulta, k=k)
        
        if not lineasFormateadas or lineasFormateadas[0].startswith("No se encontraron resultados relevantes"):
            listaResultados.append(ResultadoBusqueda("✗ No se encontraron resultados."))
            self.notify("No se encontraron resultados para la búsqueda", severity="warning")
            return

        # Mostrar resultados
        self.notify(f"✓ Se encontraron {len(lineasFormateadas)} resultado(s)", severity="information")
        for linea in lineasFormateadas:
            listaResultados.append(ResultadoBusqueda(linea))

    def on_resultado_busqueda_seleccionado(self, evento: ResultadoBusqueda.Seleccionado) -> None:
        """Maneja cuando un resultado es clicado y muestra el documento completo."""
        try:
            # Extraer el ID del documento del texto del resultado
            # Formato: "Doc 3 — score: 0.1234 | Vista previa..." o "Doc 5: Vista previa..."
            textoResultado = evento.texto.strip()
            
            # Parsear ID del documento
            idDocumento = None
            if textoResultado.startswith("Doc "):
                partes = textoResultado.split()
                if len(partes) > 1:
                    try:
                        idDocumento = int(partes[1].strip("—:")) # Limpiar caracteres de puntuación si es necesario
                    except ValueError:
                        pass
            
            if idDocumento is None:
                self.notify(f"✗ No se pudo obtener el ID del documento desde: {textoResultado}", severity="error")
                return
            
            # Obtener corpus y recuperar documento
            corpus = obtenerCorpus()
            if not corpus.estaCargado():
                self.notify("✗ El Corpus no está cargado", severity="error")
                return
            
            # Usamos el método traducido:
            documento = corpus.obtenerDocumento(idDocumento)
            if not documento:
                self.notify(f"✗ Documento {idDocumento} no encontrado en el corpus", severity="error")
                return
            
            # Formatear y mostrar el documento
            presentacionDocumento = self.formatearDocumento(documento, idDocumento)
            self.notify(f"📄 Documento {idDocumento} abierto", severity="information")
            
            # Limpiar resultados y mostrar documento
            listaResultados: ListView = self.query_one("#lista_resultados")
            listaResultados.clear()
            
            # Mostrar contenido del documento como elementos de lista
            for linea in presentacionDocumento:
                listaResultados.append(ResultadoBusqueda(linea))
                
        except Exception as e:
            self.notify(f"✗ Error al mostrar el documento: {str(e)}", severity="error")

    def formatearDocumento(self, documento: dict, idDocumento: int) -> List[str]:
        """
        Formatea el documento para su visualización.
        
        Args:
            documento: Diccionario de documento del corpus.
            idDocumento: ID del documento.
            
        Returns:
            list: Líneas formateadas para la visualización.
        """
        lineas = []
        separador = "="*60
        lineas.append(separador)
        lineas.append(f"ID del Documento: {idDocumento}")
        lineas.append(separador)
        
        # Mostrar cada campo
        for clave, valor in documento.items():
            if valor and isinstance(valor, str):
                # Limitar ancho del campo para la visualización
                valorStr = str(valor).replace("\n", " ").strip()
                if len(valorStr) > 200:
                    valorStr = valorStr[:200] + "..."
                lineas.append(f"[{clave}]: {valorStr}")
        
        lineas.append(separador)
        lineas.append("Presiona 'Buscar' para volver a la búsqueda.")
        
        return lineas


if __name__ == "__main__":
    Camaleon().run()