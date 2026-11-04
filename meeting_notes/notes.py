import re
import textwrap
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


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

    def build_html_notes(self, markdown_notes: str, frames: list[dict], info: dict, model: str) -> str:
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
  <title>{title} - Meeting Notes</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 820px;
            margin: 2rem auto; padding: 0 1.5rem; line-height: 1.75; color: #1a1a1a; }}
    h1   {{ font-size: 24px; font-weight: 600; margin-bottom: 4px; }}
    h2   {{ font-size: 16px; font-weight: 600; margin: 1.75rem 0 6px;
            text-transform: uppercase; letter-spacing: .05em; color: #444; }}
    .meta  {{ font-size: 12px; color: #888; margin-bottom: 1.75rem;
              padding-bottom: 1rem; border-bottom: 1px solid #eee; }}
    ul   {{ padding-left: 1.4rem; }} li {{ margin-bottom: 4px; }}
    blockquote {{ border-left: 3px solid #ddd; margin: 1rem 0;
                  padding: .5rem 1rem; color: #555; background: #fafafa; }}
    strong {{ font-weight: 600; }}
    .screenshot {{ margin: 1.25rem 0; border-radius: 8px; overflow: hidden;
                   border: 1px solid #e5e5e5; }}
    .screenshot img {{ width: 100%; max-height: 260px; object-fit: cover; display: block; }}
    .screenshot figcaption {{ font-size: 12px; color: #888; padding: 6px 12px;
                               background: #f8f8f8; }}
    p {{ margin-bottom: .75rem; }}
  </style>
</head>
<body>
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
