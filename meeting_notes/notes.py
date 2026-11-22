import re
import textwrap
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from meeting_notes.themes import (
    DEFAULT_THEME,
    THEME_ATTR,
    build_export_css,
    build_export_picker_html,
    get_theme,
)


class NotesService:
    STYLE_GUIDES = {
        "Executive Summary": "Create a concise executive summary with key decisions, outcomes, and 3-5 action items.",
        "Detailed Notes": "Create comprehensive detailed notes covering all discussion points, context, and decisions.",
        "Action Items Focus": "Focus primarily on action items, owners, and next steps, grouped by priority.",
        "Technical Minutes": "Create formal meeting minutes with agenda items, motions, decisions, and technical details.",
    }

    @staticmethod
    def inline_markdown(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        return text

    def markdown_to_html(self, text: str) -> str:
        lines = text.split("\n")
        output = []
        in_list = False

        for line in lines:
            if line.strip().startswith("<figure"):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(line)
                continue

            if line.startswith("### "):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("## "):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("# "):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(f"<h2>{line[2:]}</h2>")
            elif line.startswith("> "):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                content = self.inline_markdown(line[2:])
                output.append(f"<blockquote>{content}</blockquote>")
            elif re.match(r"^[-*] ", line) or re.match(r"^\d+\. ", line):
                if not in_list:
                    output.append("<ul>")
                    in_list = True
                content = re.sub(r"^[-*] ", "", line)
                content = re.sub(r"^\d+\. ", "", content)
                output.append(f"  <li>{self.inline_markdown(content)}</li>")
            elif line.strip() == "":
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append("")
            else:
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(f"<p>{self.inline_markdown(line)}</p>")

        if in_list:
            output.append("</ul>")
        return "\n".join(output)

    def build_markdown_notes(
        self,
        markdown_notes: str,
        frames: list[dict],
        info: dict,
        images_dir: Path,
        markdown_dir: Path,
    ) -> str:
        """Save screenshots to disk and replace @@SCREENSHOT_N@@ placeholders with image links."""
        stem = info["filename"].rsplit(".", 1)[0]
        shot_dir = images_dir / stem
        shot_dir.mkdir(parents=True, exist_ok=True)

        body = markdown_notes
        for i, frame in enumerate(frames):
            placeholder = f"@@SCREENSHOT_{i + 1}@@"
            image_path = shot_dir / f"screenshot_{i + 1:02d}.jpg"
            frame["pil_image"].save(image_path, format="JPEG", quality=72)

            rel_path = Path(image_path).relative_to(markdown_dir).as_posix()
            rel_path = quote(rel_path)
            caption = f"Screenshot at {frame['time_str']}"
            img_md = f"![{caption}]({rel_path})\n\n*{caption}*"
            body = body.replace(placeholder, img_md)

        # Remove any leftover placeholders the model emitted without a matching frame.
        body = re.sub(r"@@SCREENSHOT_\d+@@", "", body)
        return body

    def build_html_notes(
        self,
        markdown_notes: str,
        frames: list[dict],
        info: dict,
        model: str,
        theme: str = DEFAULT_THEME,
    ) -> str:
        theme_id = get_theme(theme).id
        title = info["filename"].rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
        now = datetime.now().strftime("%B %d, %Y %H:%M")

        html_body = markdown_notes
        for i, frame in enumerate(frames):
            placeholder = f"@@SCREENSHOT_{i + 1}@@"
            img_tag = (
                f'<figure class="screenshot">'
                f'<img src="data:image/jpeg;base64,{frame["base64_jpeg"]}" '
                f'alt="Screenshot at {frame["time_str"]}">'
                f'<figcaption>Screenshot at {frame["time_str"]}</figcaption>'
                f"</figure>"
            )
            html_body = html_body.replace(placeholder, img_tag)

        html_body = re.sub(r"@@SCREENSHOT_\d+@@", "", html_body)
        html_body = self.markdown_to_html(html_body)

        return textwrap.dedent(
            f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{title} - Meeting Notes</title>
  <style>{build_export_css()}</style>
</head>
<body {THEME_ATTR}="{theme_id}">
  {build_export_picker_html(theme_id)}
  <h1>{title}</h1>
  <div class="meta">
    Generated {now} &nbsp;.&nbsp; Model: {model}
    &nbsp;.&nbsp; Duration: {info['duration_str']}
    &nbsp;.&nbsp; {len(frames)} screenshots
  </div>
  {html_body}
</body>
</html>"""
        )
