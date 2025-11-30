from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Input, ListItem, ListView, Label, Static
from textual.message import Message


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
        yield Input(placeholder="Buscar...", id="search_input")
        yield Button("Buscar", id="search_button")
        yield ListView(id="results_list")


class Perplexity(App):

    CSS = """
    #results_list {
        height: 1fr;
        border: solid green;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Browser()
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Cuando se presiona el botón de buscar."""
        if event.button.id == "search_button":
            self.run_search()

    def run_search(self):
        """Simula resultados."""
        results_list: ListView = self.query_one("#results_list")

        results_list.clear()  # Limpia resultados anteriores

        for i in range(5):
            text = f"Resultado #{i+1}: este es el contenido del resultado."
            results_list.append(SearchResult(text))

    def on_search_result_selected(self, event: SearchResult.Selected) -> None:
        """Captura cuando un resultado es clicado."""
        self.notify(f"Clic en: {event.text}")


if __name__ == "__main__":
    Perplexity().run()
