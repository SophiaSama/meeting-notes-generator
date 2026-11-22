"""Art-style visual themes shared by the live UI and exported HTML notes.

Themes are "inspired by" named art styles: palettes, typography and simple CSS
motifs only. No characters, logos or copyrighted artwork are used.
"""

from dataclasses import dataclass

DEFAULT_THEME = "default"
THEME_ATTR = "data-art-theme"
THEME_STORAGE_KEY = "meeting-notes-art-theme"


@dataclass(frozen=True)
class ArtTheme:
    id: str
    label: str
    description: str
    font: str  # CSS font-family stack (system fallbacks, works offline)
    radius: str
    # Light palette
    bg: str
    surface: str
    fg: str
    muted: str
    accent: str
    on_accent: str
    border: str
    # Dark palette (UI only)
    dark_bg: str
    dark_surface: str
    dark_fg: str
    dark_muted: str
    # Decorative CSS applied to the page background
    background_image: str = "none"
    heading_transform: str = "none"
    heading_color: str = ""  # falls back to fg


_SERIF_CJK = '"Songti SC", "Noto Serif SC", "STKaiti", "Kaiti SC", Georgia, serif'

THEMES: dict[str, ArtTheme] = {
    t.id: t
    for t in [
        ArtTheme(
            id=DEFAULT_THEME,
            label="Default",
            description="Clean neutral look.",
            font='system-ui, -apple-system, "Segoe UI", sans-serif',
            radius="8px",
            bg="#ffffff", surface="#f8f8f8", fg="#1a1a1a", muted="#666666",
            accent="#2563eb", on_accent="#ffffff", border="#e5e5e5",
            dark_bg="#0f172a", dark_surface="#1e293b", dark_fg="#e2e8f0", dark_muted="#94a3b8",
            heading_transform="uppercase",
        ),
        ArtTheme(
            id="ink-wash",
            label="Ink Wash (Chinese painting)",
            description="Rice paper, brush black, a single vermilion seal.",
            font=_SERIF_CJK,
            radius="2px",
            bg="#f4efe6", surface="#ebe4d6", fg="#1c1c1c", muted="#5b5750",
            accent="#b3261e", on_accent="#ffffff", border="#cfc6b4",
            dark_bg="#15171a", dark_surface="#1f2226", dark_fg="#e8e4da", dark_muted="#a39e92",
            background_image="linear-gradient(180deg, rgba(60,60,60,.10) 0%, rgba(60,60,60,0) 28%)",
            heading_color="#1c1c1c",
        ),
        ArtTheme(
            id="ukiyo-e",
            label="Ukiyo-e (Japanese woodblock)",
            description="Indigo, cream and muted red with wave motifs.",
            font='"Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", Georgia, serif',
            radius="3px",
            bg="#f3ead3", surface="#e9dcb9", fg="#14233a", muted="#4a5568",
            accent="#1f3a5f", on_accent="#ffffff", border="#1f3a5f",
            dark_bg="#0e1a2b", dark_surface="#162539", dark_fg="#f0e6cc", dark_muted="#a8b3c4",
            background_image=(
                "radial-gradient(circle at 100% 100%, transparent 0 14px, rgba(31,58,95,.18) 15px 17px, "
                "transparent 18px 26px, rgba(31,58,95,.18) 27px 29px, transparent 30px)"
            ),
            heading_color="#1f3a5f",
        ),
        ArtTheme(
            id="rococo",
            label="Rococo (French)",
            description="Pastel pink and mint with gilded accents.",
            font='"Cormorant Garamond", "Playfair Display", "Palatino Linotype", Georgia, serif',
            radius="18px",
            bg="#fdf3f1", surface="#f8e4e4", fg="#4a3340", muted="#7d6672",
            accent="#9c6b1f", on_accent="#ffffff", border="#d9b26a",
            dark_bg="#2a1f26", dark_surface="#392a33", dark_fg="#f6e6e2", dark_muted="#c9b0b8",
            background_image="linear-gradient(135deg, rgba(168,222,205,.30), rgba(244,184,196,.30))",
            heading_color="#9c6b1f",
        ),
        ArtTheme(
            id="impressionism",
            label="Impressionism",
            description="Soft dappled light: lilac, water-lily green, warm yellow.",
            font='"Iowan Old Style", "Palatino Linotype", Georgia, serif',
            radius="14px",
            bg="#f1f5f9", surface="#e4ecf3", fg="#2b3a4a", muted="#5d6f80",
            accent="#4f7cac", on_accent="#ffffff", border="#c7d6e3",
            dark_bg="#1a2430", dark_surface="#243242", dark_fg="#e6edf3", dark_muted="#9fb2c4",
            background_image=(
                "radial-gradient(ellipse at 15% 10%, rgba(176,153,214,.35), transparent 45%),"
                "radial-gradient(ellipse at 85% 20%, rgba(250,222,130,.35), transparent 40%),"
                "radial-gradient(ellipse at 50% 100%, rgba(130,190,160,.35), transparent 50%)"
            ),
            heading_color="#3d5a80",
        ),
        ArtTheme(
            id="ghibli-inspired",
            label="Hand-drawn Meadow (Miyazaki-inspired)",
            description="Sky blue, meadow green and warm cream; soft and rounded.",
            font='"Nunito", "Avenir Next", "Trebuchet MS", system-ui, sans-serif',
            radius="22px",
            bg="#fbf6e6", surface="#e6f2dd", fg="#27402e", muted="#52695a",
            accent="#2f7d4f", on_accent="#ffffff", border="#a9cfa0",
            dark_bg="#14231c", dark_surface="#1d3328", dark_fg="#e8f2e1", dark_muted="#a3bfaa",
            background_image=(
                "linear-gradient(180deg, rgba(135,200,240,.45) 0%, rgba(135,200,240,0) 35%),"
                "radial-gradient(ellipse at 20% 6%, rgba(255,255,255,.9) 0 6%, transparent 7%),"
                "radial-gradient(ellipse at 28% 8%, rgba(255,255,255,.9) 0 5%, transparent 6%)"
            ),
            heading_color="#2f7d4f",
        ),
        ArtTheme(
            id="springfield-inspired",
            label="Yellow Cartoon (Simpsons-inspired)",
            description="Bright yellow, sky blue, bold black outlines.",
            font='"Comic Sans MS", "Chalkboard SE", "Marker Felt", system-ui, sans-serif',
            radius="12px",
            bg="#9fd3f5", surface="#fed41d", fg="#111111", muted="#2d2d2d",
            accent="#1d5fa8", on_accent="#ffffff", border="#111111",
            dark_bg="#0e2a45", dark_surface="#3a3200", dark_fg="#fff4b8", dark_muted="#e6d98a",
            background_image=(
                "radial-gradient(ellipse at 20% 12%, rgba(255,255,255,.95) 0 7%, transparent 8%),"
                "radial-gradient(ellipse at 28% 14%, rgba(255,255,255,.95) 0 6%, transparent 7%),"
                "radial-gradient(ellipse at 75% 8%, rgba(255,255,255,.95) 0 8%, transparent 9%)"
            ),
            heading_color="#111111",
        ),
    ]
}


def theme_choices() -> list[tuple[str, str]]:
    """(label, id) pairs for a dropdown."""
    return [(t.label, t.id) for t in THEMES.values()]


def get_theme(theme_id: str | None) -> ArtTheme:
    return THEMES.get(theme_id or DEFAULT_THEME, THEMES[DEFAULT_THEME])


def _ui_block(t: ArtTheme) -> str:
    sel = f'[{THEME_ATTR}="{t.id}"]'
    light = f"""
html{sel}, body{sel}, body{sel} .gradio-container {{
  --body-background-fill: {t.bg};
  --background-fill-primary: {t.surface};
  --background-fill-secondary: {t.bg};
  --body-text-color: {t.fg};
  --body-text-color-subdued: {t.muted};
  --block-title-text-color: {t.heading_color or t.fg};
  --block-label-text-color: {t.muted};
  --border-color-primary: {t.border};
  --block-border-color: {t.border};
  --color-accent: {t.accent};
  --button-primary-background-fill: {t.accent};
  --button-primary-background-fill-hover: {t.accent};
  --button-primary-text-color: {t.on_accent};
  --block-radius: {t.radius};
  --button-large-radius: {t.radius};
  --button-small-radius: {t.radius};
  --font: {t.font};
  font-family: {t.font};
}}
body{sel} .gradio-container {{ background-image: {t.background_image}; background-attachment: fixed; }}
body{sel} h1, body{sel} h2, body{sel} h3 {{ color: {t.heading_color or t.fg}; }}
"""
    dark = f"""
.dark{sel}, .dark body{sel}, body.dark{sel}, .dark{sel} .gradio-container, body.dark{sel} .gradio-container {{
  --body-background-fill: {t.dark_bg};
  --background-fill-primary: {t.dark_surface};
  --background-fill-secondary: {t.dark_bg};
  --body-text-color: {t.dark_fg};
  --body-text-color-subdued: {t.dark_muted};
  --block-title-text-color: {t.dark_fg};
}}
.dark body{sel} h1, body.dark{sel} h1, .dark body{sel} h2, body.dark{sel} h2, .dark body{sel} h3, body.dark{sel} h3 {{ color: {t.dark_fg}; }}
"""
    return light + dark


def build_ui_css() -> str:
    """CSS for the live Gradio UI. Themes activate via body[data-art-theme]."""
    return "\n".join(_ui_block(t) for t in THEMES.values() if t.id != DEFAULT_THEME)


def _export_block(t: ArtTheme) -> str:
    sel = f'body[{THEME_ATTR}="{t.id}"]'
    return f"""
{sel} {{ --bg: {t.bg}; --surface: {t.surface}; --fg: {t.fg}; --muted: {t.muted};
  --accent: {t.accent}; --border: {t.border}; --radius: {t.radius};
  --heading: {t.heading_color or t.fg}; --heading-transform: {t.heading_transform};
  --font: {t.font}; --bg-image: {t.background_image}; }}"""


def build_export_css() -> str:
    """Self-contained CSS for exported HTML notes (variables + all theme blocks)."""
    d = THEMES[DEFAULT_THEME]
    base = f"""
    body {{ --bg: {d.bg}; --surface: {d.surface}; --fg: {d.fg}; --muted: {d.muted};
      --accent: {d.accent}; --border: {d.border}; --radius: {d.radius};
      --heading: #444; --heading-transform: uppercase; --font: {d.font}; --bg-image: none;
      font-family: var(--font); color: var(--fg); background: var(--bg);
      background-image: var(--bg-image); background-attachment: fixed;
      max-width: 820px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.75; }}
    h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 4px; color: var(--heading); }}
    h2 {{ font-size: 16px; font-weight: 600; margin: 1.75rem 0 6px; letter-spacing: .05em;
          text-transform: var(--heading-transform); color: var(--heading); }}
    .meta {{ font-size: 12px; color: var(--muted); margin-bottom: 1.75rem;
             padding-bottom: 1rem; border-bottom: 1px solid var(--border); }}
    ul {{ padding-left: 1.4rem; }} li {{ margin-bottom: 4px; }}
    blockquote {{ border-left: 3px solid var(--accent); margin: 1rem 0; padding: .5rem 1rem;
                  color: var(--muted); background: var(--surface); border-radius: var(--radius); }}
    strong {{ font-weight: 600; }}
    .screenshot {{ margin: 1.25rem 0; border-radius: var(--radius); overflow: hidden;
                   border: 1px solid var(--border); }}
    .screenshot img {{ width: 100%; max-height: 260px; object-fit: cover; display: block; }}
    .screenshot figcaption {{ font-size: 12px; color: var(--muted); padding: 6px 12px;
                              background: var(--surface); }}
    p {{ margin-bottom: .75rem; }}
    .theme-picker {{ float: right; font-size: 12px; }}
    .theme-picker select {{ font: inherit; color: var(--fg); background: var(--surface);
                            border: 1px solid var(--border); border-radius: var(--radius); }}
    @media print {{ .theme-picker {{ display: none; }} }}
"""
    return base + "".join(_export_block(t) for t in THEMES.values() if t.id != DEFAULT_THEME)


def build_export_picker_html(selected: str) -> str:
    """A <select> that switches themes in an exported page (inline JS, no deps)."""
    options = "".join(
        f'<option value="{t.id}"{" selected" if t.id == selected else ""}>{t.label}</option>'
        for t in THEMES.values()
    )
    return (
        '<label class="theme-picker">Theme '
        f'<select onchange="document.body.setAttribute(\'{THEME_ATTR}\', this.value)">'
        f"{options}</select></label>"
    )


# Applies a theme id to <body> and remembers it in localStorage.
APPLY_THEME_JS = f"""(id) => {{
    const theme = id || 'default';
    document.body.setAttribute('{THEME_ATTR}', theme);
    document.documentElement.setAttribute('{THEME_ATTR}', theme);
    try {{ localStorage.setItem('{THEME_STORAGE_KEY}', theme); }} catch (e) {{}}
    return theme;
}}"""

# Restores the saved theme on page load and syncs the dropdown value.
RESTORE_THEME_JS = f"""() => {{
    let theme = 'default';
    try {{ theme = localStorage.getItem('{THEME_STORAGE_KEY}') || 'default'; }} catch (e) {{}}
    document.body.setAttribute('{THEME_ATTR}', theme);
    document.documentElement.setAttribute('{THEME_ATTR}', theme);
    return theme;
}}"""
