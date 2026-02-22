import os
import sys
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
    
    # Assertions
    assert not at.exception, f"App crashed with exception: {at.exception}"
    
    # Initial Sidebar check
    # Check if title contains "Agentic Lab" or known text
    # Streamlit testing API is limited for complex HTML/Markdown parsing,
    # but we can check if elements are rendered.
    
    # Check if sidebar image loaded (indicates sidebar rendered)
    # len(at.sidebar.image) > 0
    
    # Check if tabs are rendered in main area
    # len(at.tabs) >= 2
