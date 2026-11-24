"""Run the four Streamlit pages against the committed data and model."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def check_page(app):
    assert not app.exception, [e.message for e in app.exception]
    assert not app.error, [e.value for e in app.error]


if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(entry), default_timeout=60).run()
    check_page(app)
    for page in ("Аналитика", "Прогноз", "Модель"):
        app.sidebar.radio[0].set_value(page).run()
        check_page(app)
        if page == "Прогноз":
            scenario = app.selectbox[-1]
            for option in scenario.options:
                app.selectbox[-1].set_value(option).run()
                check_page(app)
                assert app.metric[0].value.endswith("чел.")
        if page == "Модель":
            assert any("GradientBoostingRegressor" in item.value for item in app.caption)
    print("OK: four pages, fitted model and four what-if scenarios")
