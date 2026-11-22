## Problem Statement

Meeting Notes Helper has one look: a blue-accent default Gradio theme with a light/dark toggle, and a plain grey exported HTML page. Users who share Meeting Notes, or who spend hours in the app, cannot give either a personal or expressive look. The first attempt at art-style themes also showed that a light palette cannot simply be reused in dark mode: accents, borders, and decorative backgrounds tuned for light looked wrong when the page went dark.

## Solution

Users pick a **Visual Theme** from a dropdown beside the existing Light/Dark button. Six art-inspired themes are offered besides Default: Ink Wash (Chinese painting), Ukiyo-e (Japanese woodblock), Rococo (French), Impressionism, a hand-drawn meadow look inspired by Miyazaki films, and a yellow-cartoon look inspired by The Simpsons. Each theme has a dedicated light palette and a dedicated dark palette designed for that style (for example, a moonlit ink scroll or a starry forest), not an inversion of the light one. The choice applies instantly, persists in the browser, and works together with Light/Dark. Exported HTML Meeting Notes carry the same themes: a picker and an Auto/Light/Dark selector are embedded in the file, so a recipient can restyle it without the app. Auto follows the recipient's system dark setting.

Miyazaki and Simpsons themes use palette and typography only, with "inspired" in the name. No characters, logos, or artwork are used.

## User Stories

1. As a user, I want a Visual Theme dropdown in the title row, so that I can restyle the app without opening settings.
2. As a user, I want the theme to change instantly with no reload, so that I can compare styles quickly.
3. As a user, I want my chosen theme remembered in my browser, so that the app opens in my style next time.
4. As a user, I want Visual Theme and Light/Dark to be independent, so that I can combine any style with either mode.
5. As a user who likes ink painting, I want an Ink Wash theme with rice-paper tones and a single red accent, so that the app feels like a scroll.
6. As a user, I want an Ukiyo-e theme with indigo, cream, and wave motifs, so that I get a woodblock-print feel.
7. As a user, I want a Rococo theme with pastel and gilt tones and rounded ornate styling, so that I get a French salon feel.
8. As a user, I want an Impressionism theme with soft, dappled colour glows, so that the page feels like a painting rather than a form.
9. As a Miyazaki fan, I want a soft, rounded meadow theme with sky and cloud motifs, so that the app feels hand-drawn and calm.
10. As a Simpsons fan, I want a bright yellow and sky-blue cartoon theme with bold outlines, so that the app feels playful.
11. As a user in dark mode, I want each theme to have its own dark palette, so that the style survives at night instead of looking like a washed-out light theme.
12. As a user in dark mode, I want the accent, border, and background motif of each theme re-tuned for dark, so that nothing glares or disappears.
13. As a user, I want text, muted text, headings, and button labels to stay readable in every theme and both modes, so that style never costs legibility.
14. As a user, I want inputs, secondary buttons, panels, and placeholders to follow the theme, so that no light-mode widgets remain in dark mode.
15. As a user, I want Default to stay available and unchanged in look, so that I can opt out of art styles.
16. As a user who shares Meeting Notes, I want the exported HTML to carry a theme picker, so that the recipient can choose a style.
17. As a recipient of an exported file, I want an Auto/Light/Dark selector that defaults to my system setting, so that the page matches my environment.
18. As a user, I want the export to remain one self-contained file with no external fonts or scripts, so that it works offline and can be emailed.
19. As a user printing the notes, I want the picker and selectors hidden in print, so that the printout is clean.
20. As a user of the MCP or pipeline, I want exports to default to Default theme when no theme is given, so that existing behaviour does not change.
21. As a user, I want an unknown or stale saved theme id to fall back to Default, so that the app never breaks on a bad value.
22. As a maintainer, I want to add a new theme by adding one registry entry with a light and a dark palette, so that the UI and the export both pick it up.
23. As a keyboard user, I want the Visual Theme dropdown to be labelled and focusable, so that I can change it without a mouse.

## Implementation Decisions

- **One registry is the source of truth.** Each theme defines an id, display label, description, font stack, corner radius, heading case, and two palettes (light, dark). A palette holds: page background, surface (cards/panels), text, muted text, accent, text-on-accent, border, heading colour, and a decorative background image (CSS gradients only). The live UI CSS and the exported-HTML CSS are both generated from this registry.
- **Dark palettes are authored, not derived.** Each dark palette is designed for its style (moonlit ink, indigo night with gold, midnight plum with gilt, Monet twilight, starry forest, navy sky with yellow outlines). Accent, border, heading, and background motif are all re-chosen for dark.
- **Live UI activation.** The selected theme id is set as an attribute on the page root and body. The generated CSS overrides Gradio's theme variables for the chosen theme, once for light and once for dark, keyed off the existing `dark` class. Variables cover backgrounds, panels, inputs, secondary and primary buttons, borders, text, placeholders, accent, radius, and font, so no widget keeps a light-mode value in dark mode.
- **Independence from Light/Dark.** The existing Light/Dark toggle and its storage key are unchanged. Visual Theme uses its own browser storage key and is restored on page load, which also sets the dropdown value.
- **Dropdown placement.** The Visual Theme dropdown sits in the title row beside the Light/Dark button, without a visible label and with an accessible label.
- **Fallback.** Any unknown theme id (saved, passed in, or exported) resolves to Default.
- **Exported HTML.** The notes builder accepts an optional theme id (default: Default). The exported file embeds CSS for all themes in both modes, a theme select, and a mode select (Auto, Light, Dark). Auto uses the recipient's `prefers-color-scheme`; Light and Dark override it. Switching uses small inline scripts only. Fonts are system and common-web font stacks with generic fallbacks; nothing is fetched. The pickers are hidden when printing.
- **No pipeline plumbing in this change.** The app's selected Visual Theme is not passed into generation, so exports begin on Default and are switched with the embedded picker. Passing it through is a later, separate change because the pipeline handlers use long positional signatures.
- **Naming and legal.** Miyazaki and Simpsons themes are labelled "inspired", use no characters, logos, or artwork, and use palette and typography only.
- **Accessibility floor.** In every theme and mode: body text on page background and on surface, and muted text on surface, meet 4.5:1; button label on accent meets 4.5:1; headings on page background meet 3:1.
- **Delivery order, one reviewable change each:** (1) registry and export CSS, (2) live UI dropdown and light palettes, (3) dedicated dark palettes and the extra widget variables, (4) export Auto/Light/Dark selector.

## Testing Decisions

- A good test checks external behaviour at the interface, not CSS text layout or colour literals beyond the stated rules.
- **Seam 1, theme registry and generated CSS (pure functions).** Tests: all six requested themes plus Default exist; unknown or empty id falls back to Default; every theme has both palettes and the dark palette differs from the light one; dark backgrounds are dark; contrast floors above hold for every theme and both modes; the UI CSS contains a dark rule for each theme; the export CSS supports the dark media query and the explicit dark override.
- **Seam 2, notes builder output.** Tests: the export applies the requested theme on the body; selects the matching option in the picker; defaults to Default; ignores an unknown theme; includes the mode selector.
- Visual quality in a real browser (each theme, light and dark, narrow window, dropdown popups, chat and table widgets) is verified manually. An existing Playwright theme test is skipped when Playwright is not installed; extending it to cycle themes is optional follow-up.
- Prior art: the pure view-builder tests in the views test module and the notes-builder usage in the end-to-end pipeline tests.

## Out of Scope

- Passing the app's Visual Theme into generation or the MCP server tools.
- AI-generated cover art or illustrations, and restyling Screenshots.
- Writing the Meeting Notes in a themed voice.
- Embedded web fonts, and custom user-defined themes through the UI.
- Using real characters, logos, or artwork from Miyazaki films or The Simpsons.
- Changes to Notes Style, pipeline, LLM, or video behaviour.

## Further Notes

- Vocabulary follows `GLOSSARY.md`. "Visual Theme" is distinct from Notes Style: Notes Style changes what the Meeting Notes say, Visual Theme changes how they look. This term is not yet in the glossary; consider adding it.
- Known risks to check in a browser: the yellow and pastel themes for glare and legibility; popup menus and other Gradio widgets that may use variables not overridden; fonts that exist only on some machines.
- Builds on `specs/polished-ui.md`, whose theme and dark-mode work is the baseline.
