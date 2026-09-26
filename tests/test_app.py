from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "artifacts" / "metadata.json").exists(), reason="Generate artifacts first")
def test_dashboard_navigation_and_controls():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    for section in ["Model comparison", "Forecast horizons", "Statistical evidence", "Method & interview"]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception, section
        if section == "Model comparison":
            app.selectbox[0].set_value(20).run()
            app.multiselect[0].set_value([]).run()
            assert not app.exception
        elif section == "Forecast horizons":
            app.selectbox[0].set_value("Exponential Smoothing").run()
            app.selectbox[1].set_value(20).run()
            assert not app.exception
