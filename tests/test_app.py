"""Smoke test Streamlit aplikácie (bez prehliadača)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def test_app_starts_without_exception() -> None:
    app = _run_app()
    assert not app.exception, [exception.value for exception in app.exception]
    assert len(app.metric) >= 3


def test_app_supports_model_and_team_change() -> None:
    app = _run_app()
    app.selectbox[2].set_value("gradient_boosting").run()
    assert not app.exception, [exception.value for exception in app.exception]

    # Vyberieme posledný dostupný domáci tím zo zoznamu aktuálnej sezóny
    target_team = app.selectbox[0].options[-1]
    app.selectbox[0].set_value(target_team).run()
    assert not app.exception, [exception.value for exception in app.exception]
    assert app.metric[0].value.endswith("%")
