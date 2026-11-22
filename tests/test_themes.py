from meeting_notes.notes import NotesService
from meeting_notes.themes import (
    DEFAULT_THEME,
    THEMES,
    build_export_css,
    build_ui_css,
    get_theme,
    theme_choices,
)

INFO = {"filename": "weekly_sync.mp4", "duration_str": "10:00"}
EXPECTED = {"ink-wash", "ukiyo-e", "rococo", "impressionism", "ghibli-inspired", "springfield-inspired"}


def test_registry_contains_requested_styles():
    assert EXPECTED <= set(THEMES)
    assert DEFAULT_THEME in THEMES
    assert len(theme_choices()) == len(THEMES)


def test_get_theme_falls_back_to_default():
    assert get_theme("nope").id == DEFAULT_THEME
    assert get_theme(None).id == DEFAULT_THEME


def test_every_theme_has_ui_and_export_css():
    ui, export = build_ui_css(), build_export_css()
    for tid in EXPECTED:
        assert f'data-art-theme="{tid}"' in ui
        assert f'data-art-theme="{tid}"' in export


def test_html_export_applies_selected_theme():
    html = NotesService().build_html_notes("## Hi", [], INFO, "m", theme="ink-wash")
    assert '<body data-art-theme="ink-wash">' in html
    assert '<option value="ink-wash" selected>' in html


def test_html_export_defaults_and_ignores_unknown_theme():
    svc = NotesService()
    assert f'data-art-theme="{DEFAULT_THEME}"' in svc.build_html_notes("x", [], INFO, "m")
    assert f'data-art-theme="{DEFAULT_THEME}"' in svc.build_html_notes("x", [], INFO, "m", theme="bogus")
