from meeting_notes.notes import NotesService
from meeting_notes.themes import (
    DEFAULT_THEME,
    FONT_IMPORT,
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


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_dark_palettes_are_dedicated_not_copies():
    for t in THEMES.values():
        assert t.dark != t.light
        assert _lum(t.dark.bg) < 0.1, f"{t.id} dark bg should be dark"
        assert _lum(t.dark.fg) > _lum(t.dark.bg)


def test_palettes_meet_contrast_in_both_modes():
    for t in THEMES.values():
        for mode, p in (("light", t.light), ("dark", t.dark)):
            assert _contrast(p.fg, p.bg) >= 4.5, (t.id, mode, "fg/bg")
            assert _contrast(p.fg, p.surface) >= 4.5, (t.id, mode, "fg/surface")
            assert _contrast(p.muted, p.surface) >= 4.5, (t.id, mode, "muted/surface")
            assert _contrast(p.on_accent, p.accent) >= 4.5, (t.id, mode, "button")
            assert _contrast(p.heading, p.bg) >= 3.0, (t.id, mode, "heading/bg")


def test_ui_css_has_dark_rules_per_theme():
    ui = build_ui_css()
    for tid in EXPECTED:
        assert f'html.dark[data-art-theme="{tid}"]' in ui


def test_export_supports_dark_via_media_query_and_override():
    css = build_export_css()
    assert "prefers-color-scheme: dark" in css
    assert 'data-mode="dark"' in css
    html = NotesService().build_html_notes("x", [], INFO, "m")
    assert 'aria-label="Mode"' in html


def test_export_declares_color_scheme_on_root():
    html = NotesService().build_html_notes("x", [], INFO, "m")
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "html { color-scheme: light dark; }" in html


# Primary (first-listed) typeface per theme.
FONT_MAP = {
    DEFAULT_THEME: "Inter",
    "ink-wash": "Noto Serif SC",
    "ukiyo-e": "Shippori Mincho",
    "rococo": "Playfair Display",
    "impressionism": "Lora",
    "ghibli-inspired": "Nunito",
    "springfield-inspired": "Comic Neue",
}


def _primary_font(font_stack: str) -> str:
    return font_stack.split(",")[0].strip().strip("\"'")


def test_font_map_covers_every_theme():
    assert set(FONT_MAP) == set(THEMES)


def test_each_theme_uses_its_assigned_primary_font():
    for tid, family in FONT_MAP.items():
        assert _primary_font(THEMES[tid].font) == family, tid


def test_each_theme_has_a_distinct_primary_font():
    fonts = [_primary_font(t.font) for t in THEMES.values()]
    assert len(set(fonts)) == len(fonts)


def test_web_fonts_are_imported_for_every_theme():
    for family in FONT_MAP.values():
        assert family.replace(" ", "+") in FONT_IMPORT, family
    assert build_ui_css().startswith("@import")
    assert build_export_css().startswith("@import")


def test_fonts_are_wired_into_ui_and_export_css():
    ui, export = build_ui_css(), build_export_css()
    for tid in FONT_MAP:
        font = THEMES[tid].font
        assert f"--font: {font}" in export, tid
        if tid != DEFAULT_THEME:  # default keeps Gradio's own styling in the live UI
            assert f"--font: {font}" in ui, tid
