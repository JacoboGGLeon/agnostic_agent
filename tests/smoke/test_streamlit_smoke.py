import os
from streamlit.testing.v1 import AppTest

def test_streamlit_smoke():
    """
    Basic smoke test to ensure the Streamlit app starts without crashing.
    Using Streamlit's AppTest framework.
    """
    # Adjust path to point to the UI entry point
    app_path = "agnostic_agent/ui/streamlit_app.py"
    
    # Check if file exists
    if not os.path.exists(app_path):
         # Try with absolute path relative to project root
         app_path = os.path.join(os.getcwd(), "agnostic_agent", "ui", "streamlit_app.py")
    
    at = AppTest.from_file(app_path)
    at.run(timeout=15)
    
    assert not at.exception, f"App crashed with exception: {at.exception}"
    assert len(at.tabs) >= 2
    assert [tab.label for tab in at.tabs[:2]] == ["Online Chat", "Offline Manager"]
    online_markdown = [getattr(item, "value", "") for item in at.tabs[0].markdown]
    assert any("Modo de trabajo" in value for value in online_markdown)
    assert len(at.chat_input) == 1
    assert getattr(at.chat_input[0], "placeholder", "") == "Escribe tu mensaje..."

    offline_tab = at.tabs[1]
    offline_markdown = [getattr(item, "value", "") for item in offline_tab.markdown]
    assert any("Gestor de Conocimiento" in value for value in offline_markdown)
    assert any("Tools Playground" in value for value in offline_markdown)
    assert any("Gestor de Skills" in value for value in offline_markdown)
