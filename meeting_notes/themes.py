"""Art-style visual themes shared by the live UI and exported HTML notes.

Themes are "inspired by" named art styles: palettes, typography and simple CSS
motifs only. No characters, logos or copyrighted artwork are used.

Every theme ships a dedicated light *and* dark palette; the dark palette is
designed per style (e.g. a moonlit ink wash, a night-forest meadow) rather than
derived by inverting the light one.
"""

from dataclasses import dataclass

DEFAULT_THEME = "default"
THEME_ATTR = "data-art-theme"
MODE_ATTR = "data-mode"
THEME_STORAGE_KEY = "meeting-notes-art-theme"


@dataclass(frozen=True)
class Palette:
    bg: str  # page background
    surface: str  # cards, blocks, panels
    fg: str  # body text
    muted: str  # secondary text
    accent: str  # primary buttons, quote bars
    on_accent: str  # text drawn on top of accent
    border: str
    heading: str
    image: str = "none"  # decorative page-background motif (CSS image list)


@dataclass(frozen=True)
class ArtTheme:
    id: str
    label: str
    description: str
    font: str  # CSS font-family stack (system fallbacks, works offline)
    radius: str
    light: Palette
    dark: Palette
    heading_transform: str = "none"


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
            heading_transform="uppercase",
            light=Palette(
                bg="#ffffff", surface="#f8f8f8", fg="#1a1a1a", muted="#666666",
                accent="#2563eb", on_accent="#ffffff", border="#e5e5e5", heading="#444444",
            ),
            dark=Palette(
                bg="#0f172a", surface="#1e293b", fg="#e2e8f0", muted="#94a3b8",
                accent="#3b6fe0", on_accent="#ffffff", border="#334155", heading="#cbd5e1",
            ),
        ),
        ArtTheme(
            id="ink-wash",
            label="Ink Wash (Chinese painting)",
            description="Rice paper and brush black; by night, a moonlit ink scroll.",
            font=_SERIF_CJK,
            radius="2px",
            light=Palette(
                bg="#f4efe6", surface="#ebe4d6", fg="#1c1c1c", muted="#5b5750",
                accent="#b3261e", on_accent="#ffffff", border="#cfc6b4", heading="#1c1c1c",
                image="linear-gradient(180deg, rgba(60,60,60,.12) 0%, rgba(60,60,60,0) 28%)",
            ),
            dark=Palette(
                bg="#1b1a18", surface="#292622", fg="#ece6da", muted="#b5aea0",
                accent="#d93829", on_accent="#ffffff", border="#524c44", heading="#f5efe2",
                image=(
                    "radial-gradient(circle at 86% 9%, rgba(245,239,226,.28) 0 4.5%, rgba(245,239,226,.08) 5%, transparent 9%),"
                    "linear-gradient(180deg, rgba(235,228,214,.14) 0%, rgba(235,228,214,0) 38%)"
                ),
            ),
        ),
        ArtTheme(
            id="ukiyo-e",
            label="Ukiyo-e (Japanese woodblock)",
            description="Indigo, cream and red waves; by night, deep indigo with a gold moon.",
            font='"Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", Georgia, serif',
            radius="3px",
            light=Palette(
                bg="#f3ead3", surface="#e9dcb9", fg="#14233a", muted="#4a5568",
                accent="#1f3a5f", on_accent="#ffffff", border="#1f3a5f", heading="#1f3a5f",
                image=(
                    "radial-gradient(circle at 100% 100%, transparent 0 14px, rgba(31,58,95,.18) 15px 17px, "
                    "transparent 18px 26px, rgba(31,58,95,.18) 27px 29px, transparent 30px)"
                ),
            ),
            dark=Palette(
                bg="#102340", surface="#1b355c", fg="#f3e9cf", muted="#b6c5dc",
                accent="#e5b54f", on_accent="#14233a", border="#e5b54f", heading="#f2cf7e",
                image=(
                    "radial-gradient(circle at 100% 100%, transparent 0 14px, rgba(160,200,255,.24) 15px 17px, "
                    "transparent 18px 26px, rgba(160,200,255,.24) 27px 29px, transparent 30px)"
                ),
            ),
        ),
        ArtTheme(
            id="rococo",
            label="Rococo (French)",
            description="Pastel pink, mint and gilt; by night, midnight plum and gold.",
            font='"Cormorant Garamond", "Playfair Display", "Palatino Linotype", Georgia, serif',
            radius="18px",
            light=Palette(
                bg="#fdf3f1", surface="#f8e4e4", fg="#4a3340", muted="#6c5460",
                accent="#9c6b1f", on_accent="#ffffff", border="#d9b26a", heading="#9c6b1f",
                image="linear-gradient(135deg, rgba(168,222,205,.32), rgba(244,184,196,.32))",
            ),
            dark=Palette(
                bg="#2e1b2b", surface="#442840", fg="#faece8", muted="#d9bec6",
                accent="#e2b857", on_accent="#2a1620", border="#c99b49", heading="#f0cf7b",
                image=(
                    "radial-gradient(ellipse at 12% 4%, rgba(220,140,175,.30), transparent 52%),"
                    "radial-gradient(ellipse at 88% 96%, rgba(120,195,170,.24), transparent 52%)"
                ),
            ),
        ),
        ArtTheme(
            id="impressionism",
            label="Impressionism",
            description="Dappled lilac, lily green and yellow; by night, a Monet twilight.",
            font='"Iowan Old Style", "Palatino Linotype", Georgia, serif',
            radius="14px",
            light=Palette(
                bg="#f1f5f9", surface="#e4ecf3", fg="#2b3a4a", muted="#4e6072",
                accent="#3f6a9a", on_accent="#ffffff", border="#c7d6e3", heading="#3d5a80",
                image=(
                    "radial-gradient(ellipse at 15% 10%, rgba(176,153,214,.35), transparent 45%),"
                    "radial-gradient(ellipse at 85% 20%, rgba(250,222,130,.35), transparent 40%),"
                    "radial-gradient(ellipse at 50% 100%, rgba(130,190,160,.35), transparent 50%)"
                ),
            ),
            dark=Palette(
                bg="#1b2340", surface="#273359", fg="#eaf0fa", muted="#b8c6e0",
                accent="#9dbcf0", on_accent="#101b33", border="#5a6fa3", heading="#cbbcf6",
                image=(
                    "radial-gradient(ellipse at 15% 10%, rgba(165,130,230,.36), transparent 46%),"
                    "radial-gradient(ellipse at 85% 18%, rgba(250,200,100,.26), transparent 40%),"
                    "radial-gradient(ellipse at 50% 100%, rgba(90,185,150,.32), transparent 52%)"
                ),
            ),
        ),
        ArtTheme(
            id="ghibli-inspired",
            label="Hand-drawn Meadow (Miyazaki-inspired)",
            description="Sky, meadow and cream; by night, a starry forest.",
            font='"Nunito", "Avenir Next", "Trebuchet MS", system-ui, sans-serif',
            radius="22px",
            light=Palette(
                bg="#fbf6e6", surface="#e6f2dd", fg="#27402e", muted="#52695a",
                accent="#2f7d4f", on_accent="#ffffff", border="#a9cfa0", heading="#2f7d4f",
                image=(
                    "linear-gradient(180deg, rgba(135,200,240,.45) 0%, rgba(135,200,240,0) 35%),"
                    "radial-gradient(ellipse at 20% 6%, rgba(255,255,255,.9) 0 6%, transparent 7%),"
                    "radial-gradient(ellipse at 28% 8%, rgba(255,255,255,.9) 0 5%, transparent 6%)"
                ),
            ),
            dark=Palette(
                bg="#112920", surface="#1c4031", fg="#eaf5e3", muted="#add0b5",
                accent="#6fcf97", on_accent="#0d2217", border="#4d916b", heading="#9be6ba",
                image=(
                    "linear-gradient(180deg, rgba(45,75,165,.55) 0%, rgba(45,75,165,0) 40%),"
                    "radial-gradient(circle at 12% 8%, rgba(255,255,230,.95) 0 .4%, transparent .7%),"
                    "radial-gradient(circle at 34% 14%, rgba(255,255,230,.85) 0 .35%, transparent .6%),"
                    "radial-gradient(circle at 63% 6%, rgba(255,255,230,.95) 0 .4%, transparent .7%),"
                    "radial-gradient(circle at 88% 16%, rgba(255,255,230,.85) 0 .35%, transparent .6%)"
                ),
            ),
        ),
        ArtTheme(
            id="springfield-inspired",
            label="Yellow Cartoon (Simpsons-inspired)",
            description="Bright yellow and sky blue; by night, navy sky with yellow outlines.",
            font='"Comic Sans MS", "Chalkboard SE", "Marker Felt", system-ui, sans-serif',
            radius="12px",
            light=Palette(
                bg="#9fd3f5", surface="#fed41d", fg="#111111", muted="#2d2d2d",
                accent="#1d5fa8", on_accent="#ffffff", border="#111111", heading="#111111",
                image=(
                    "radial-gradient(ellipse at 20% 12%, rgba(255,255,255,.95) 0 7%, transparent 8%),"
                    "radial-gradient(ellipse at 28% 14%, rgba(255,255,255,.95) 0 6%, transparent 7%),"
                    "radial-gradient(ellipse at 75% 8%, rgba(255,255,255,.95) 0 8%, transparent 9%)"
                ),
            ),
            dark=Palette(
                bg="#113259", surface="#2e2910", fg="#fff6c2", muted="#f0df84",
                accent="#fed41d", on_accent="#111111", border="#fed41d", heading="#fed41d",
                image=(
                    "radial-gradient(circle at 15% 10%, rgba(255,244,184,.95) 0 .45%, transparent .8%),"
                    "radial-gradient(circle at 40% 6%, rgba(255,244,184,.85) 0 .35%, transparent .65%),"
                    "radial-gradient(circle at 70% 12%, rgba(255,244,184,.95) 0 .45%, transparent .8%),"
                    "radial-gradient(circle at 92% 5%, rgba(255,244,184,.95) 0 2.5%, transparent 2.9%)"
                ),
            ),
        ),
    ]
}


def theme_choices() -> list[tuple[str, str]]:
    """(label, id) pairs for a dropdown."""
    return [(t.label, t.id) for t in THEMES.values()]


def get_theme(theme_id: str | None) -> ArtTheme:
    return THEMES.get(theme_id or DEFAULT_THEME, THEMES[DEFAULT_THEME])


# --------------------------------------------------------------------------- #
# Live Gradio UI
# --------------------------------------------------------------------------- #
def _ui_vars(t: ArtTheme, p: Palette) -> str:
    return f"""
  --body-background-fill: {p.bg};
  --background-fill-primary: {p.surface};
  --background-fill-secondary: {p.bg};
  --block-background-fill: {p.surface};
  --panel-background-fill: {p.surface};
  --input-background-fill: {p.bg};
  --input-background-fill-focus: {p.bg};
  --body-text-color: {p.fg};
  --body-text-color-subdued: {p.muted};
  --block-title-text-color: {p.heading};
  --block-label-text-color: {p.muted};
  --block-info-text-color: {p.muted};
  --input-placeholder-color: {p.muted};
  --border-color-primary: {p.border};
  --block-border-color: {p.border};
  --input-border-color: {p.border};
  --button-secondary-background-fill: {p.surface};
  --button-secondary-background-fill-hover: {p.bg};
  --button-secondary-text-color: {p.fg};
  --button-secondary-border-color: {p.border};
  --color-accent: {p.accent};
  --color-accent-soft: {p.surface};
  --button-primary-background-fill: {p.accent};
  --button-primary-background-fill-hover: {p.accent};
  --button-primary-text-color: {p.on_accent};
  --button-primary-border-color: {p.accent};
  --button-primary-border-color-hover: {p.accent};
  --block-radius: {t.radius};
  --button-large-radius: {t.radius};
  --button-small-radius: {t.radius};
  --font: {t.font};
  font-family: {t.font};"""


def _ui_block(t: ArtTheme) -> str:
    a = f'[{THEME_ATTR}="{t.id}"]'
    out = []
    for mode, p in (("light", t.light), ("dark", t.dark)):
        if mode == "light":
            roots = f"body{a}, body{a} .gradio-container"
            scope = f"body{a}"
        else:
            roots = f"html.dark{a}, body.dark{a}, html.dark{a} body, body.dark{a} .gradio-container"
            scope = f"html.dark{a} body, body.dark{a}"
        out.append(f"{roots} {{{_ui_vars(t, p)}\n}}")
        body_sel = f"body{a}" if mode == "light" else f"html.dark{a} body, body.dark{a}"
        out.append(f"{body_sel} {{ background-color: {p.bg}; }}")
        gc = (
            f"body{a} .gradio-container"
            if mode == "light"
            else f"html.dark{a} .gradio-container, body.dark{a} .gradio-container"
        )
        out.append(
            f"{gc} {{ background-image: {p.image}; background-attachment: fixed; }}"
        )
        heads = ", ".join(f"{scope} {h}" for h in ("h1", "h2", "h3"))
        out.append(f"{heads} {{ color: {p.heading}; }}")
    return "\n".join(out) + "\n"


def build_ui_css() -> str:
    """CSS for the live Gradio UI. Themes activate via [data-art-theme] and `.dark`."""
    return "\n".join(_ui_block(t) for t in THEMES.values() if t.id != DEFAULT_THEME)


# --------------------------------------------------------------------------- #
# Exported HTML
# --------------------------------------------------------------------------- #
def _export_vars(t: ArtTheme, p: Palette) -> str:
    return (
        f"--bg: {p.bg}; --surface: {p.surface}; --fg: {p.fg}; --muted: {p.muted}; "
        f"--accent: {p.accent}; --border: {p.border}; --heading: {p.heading}; "
        f"--bg-image: {p.image}; --radius: {t.radius}; --font: {t.font}; "
        f"--heading-transform: {t.heading_transform};"
    )


def _export_block(t: ArtTheme) -> str:
    sel = f'body[{THEME_ATTR}="{t.id}"]'
    light = _export_vars(t, t.light)
    dark = _export_vars(t, t.dark)
    return f"""
{sel} {{ {light} }}
@media (prefers-color-scheme: dark) {{
  {sel}:not([{MODE_ATTR}="light"]) {{ {dark} }}
}}
{sel}[{MODE_ATTR}="dark"] {{ {dark} }}"""


def build_export_css() -> str:
    """Self-contained CSS for exported HTML notes (variables + all theme blocks)."""
    base = """
    /* color-scheme belongs on the root so canvas, scrollbars and form controls adapt. */
    html { color-scheme: light dark; }
    html:has(body[data-mode="light"]) { color-scheme: light; }
    html:has(body[data-mode="dark"]) { color-scheme: dark; }
    body {
      font-family: var(--font); color: var(--fg); background: var(--bg);
      background-image: var(--bg-image); background-attachment: fixed;
      max-width: 820px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.75; }
    h1 { font-size: 24px; font-weight: 600; margin-bottom: 4px; color: var(--heading); }
    h2 { font-size: 16px; font-weight: 600; margin: 1.75rem 0 6px; letter-spacing: .05em;
         text-transform: var(--heading-transform); color: var(--heading); }
    .meta { font-size: 12px; color: var(--muted); margin-bottom: 1.75rem;
            padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
    ul { padding-left: 1.4rem; } li { margin-bottom: 4px; }
    blockquote { border-left: 3px solid var(--accent); margin: 1rem 0; padding: .5rem 1rem;
                 color: var(--muted); background: var(--surface); border-radius: var(--radius); }
    strong { font-weight: 600; }
    .screenshot { margin: 1.25rem 0; border-radius: var(--radius); overflow: hidden;
                  border: 1px solid var(--border); }
    .screenshot img { width: 100%; max-height: 260px; object-fit: cover; display: block; }
    .screenshot figcaption { font-size: 12px; color: var(--muted); padding: 6px 12px;
                             background: var(--surface); }
    p { margin-bottom: .75rem; }
    .theme-picker { float: right; font-size: 12px; display: flex; gap: 6px; }
    .theme-picker select { font: inherit; color: var(--fg); background: var(--surface);
                           border: 1px solid var(--muted); border-radius: var(--radius); }
    @media print { .theme-picker { display: none; } }
"""
    return base + "".join(_export_block(t) for t in THEMES.values())


def build_export_picker_html(selected: str) -> str:
    """Theme and light/dark selects for an exported page (inline JS, no deps)."""
    options = "".join(
        f'<option value="{t.id}"{" selected" if t.id == selected else ""}>{t.label}</option>'
        for t in THEMES.values()
    )
    set_theme = f"document.body.setAttribute('{THEME_ATTR}', this.value)"
    set_mode = (
        f"this.value==='auto'?document.body.removeAttribute('{MODE_ATTR}')"
        f":document.body.setAttribute('{MODE_ATTR}', this.value)"
    )
    return (
        '<div class="theme-picker">'
        f'<select aria-label="Theme" onchange="{set_theme}">{options}</select>'
        f'<select aria-label="Mode" onchange="{set_mode}">'
        '<option value="auto" selected>Auto</option>'
        '<option value="light">Light</option><option value="dark">Dark</option></select>'
        "</div>"
    )


# Applies a theme id to <body>/<html> and remembers it in localStorage.
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
