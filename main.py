from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Input, ListItem, ListView, Label, Static
from textual.message import Message

from controllers.browser_integration import ModelBrowser


class SearchResult(ListItem):
    """Elemento clicable de un resultado."""
    
    class Selected(Message):
        def __init__(self, sender: "SearchResult", text: str):
            super().__init__(sender)
            self.text = text

    def __init__(self, text: str):
        super().__init__(Label(text))
        self.text = text

    def on_click(self) -> None:
        self.post_message(self.Selected(self, self.text))


class Browser(Static):

    def compose(self) -> ComposeResult:
        # Model type selector
        yield Label("Selecciona el tipo de modelo:")
        yield Button("Binary Model", id="select_binary_button")
        yield Button("TF-IDF Model", id="select_tfidf_button")
        yield Button("BM25 Model", id="select_bm25_button")
        yield Label("", id="selected_model_label")

        # search area
        yield Input(placeholder="Buscar...", id="search_input")
        yield Button("Buscar", id="search_button")
        yield ListView(id="results_list")


class Perplexity(App):

    CSS = """
    #results_list {
        height: 1fr;
        border: solid green;
    }
    Browser {
        layout: vertical;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Browser()
        yield Footer()

    def on_mount(self) -> None:
        # Create the model browser bridge used by the UI
        self.model_browser = ModelBrowser()
        self.selected_model_type = None
        # Populate models list from models/ folder
        self.refresh_models_list()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Maneja clicks de botones."""
        if event.button.id == "search_button":
            self.run_search()
        elif event.button.id == "select_binary_button":
            self.select_model_type("binary")
        elif event.button.id == "select_tfidf_button":
            self.select_model_type("tfidf")
        elif event.button.id == "select_bm25_button":
            self.select_model_type("bm25")

    def select_model_type(self, model_type: str) -> None:
        """Selecciona un tipo de modelo (binary, tfidf, bm25) y lo carga."""
        path = self.model_browser.get_model_path(model_type)
        if not path:
            self.notify(f"✗ No se encontró modelo {model_type}")
            return

        success, message = self.model_browser.load(path)
        if success:
            self.notify(f"✓ {message}")
            self.selected_model_type = model_type
            label: Label = self.query_one("#selected_model_label")
            label.update(f"✓ {model_type.upper()} Model Loaded")
            
            results_list: ListView = self.query_one("#results_list")
            results_list.clear()
            results_list.append(SearchResult(message))
        else:
            self.notify(f"✗ {message}")
            results_list: ListView = self.query_one("#results_list")
            results_list.clear()
            results_list.append(SearchResult(f"Error: {message}"))

    def refresh_models_list(self) -> None:
        """Verifica modelos disponibles en la carpeta models/."""
        entries = self.model_browser.list_models()
        if entries:
            self.notify(f"✓ Se encontraron {len(entries)} modelos disponibles")
        else:
            self.notify("✗ No hay archivos .pkl en la carpeta models/")

    def run_search(self):
        """Ejecuta la búsqueda usando el modelo seleccionado."""
        results_list: ListView = self.query_one("#results_list")
        results_list.clear()

        search_input: Input = self.query_one("#search_input")
        query = (search_input.value or "").strip()

        if not query:
            msg = "Por favor ingresa una búsqueda"
            self.notify(msg)
            return

        if not self.model_browser.has_model():
            msg = "Por favor selecciona un modelo primero"
            self.notify(msg)
            return

        # Ejecutar búsqueda
        formatted = self.model_browser.search(query, k=5)
        
        if not formatted:
            results_list.append(SearchResult("✗ No se encontraron resultados."))
            self.notify("No se encontraron resultados para la búsqueda")
            return

        # Mostrar resultados
        self.notify(f"✓ Se encontraron {len(formatted)} resultado(s)")
        for line in formatted:
            results_list.append(SearchResult(line))

    def on_search_result_selected(self, event: SearchResult.Selected) -> None:
        """Maneja cuando un resultado es clicado."""
        self.notify(f"Seleccionado: {event.text}")


if __name__ == "__main__":
    Perplexity().run()
