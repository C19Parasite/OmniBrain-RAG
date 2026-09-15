"""
Test Suite for OmniBrain Streamlit Interface.
Verifies that streamlit_app.py initializes cleanly without errors.
"""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

def test_streamlit_app_initialization():
    """Verify Streamlit app mounts, loads backend singletons, and renders UI components."""
    app_path = Path(__file__).resolve().parent.parent / "streamlit_app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)
    assert not at.exception, f"Streamlit app threw unexpected exception: {at.exception}"
    assert len(at.sidebar) > 0, "Sidebar should contain configuration controls"
