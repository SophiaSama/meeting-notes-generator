FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install system dependencies and create a non-root user.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && chown appuser:appuser /app

# Switch to the non-root user before installing Python dependencies.
USER appuser

# Install Python dependencies first so Docker can cache this layer.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code.
COPY --chown=appuser:appuser meeting_notes/ meeting_notes/
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser mcp_server.py .

ENV MEETING_NOTES_HOST=0.0.0.0

EXPOSE 7860

# Verify the Gradio UI is responding.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default: run the Gradio UI. Override with e.g. `uv run python mcp_server.py`
CMD ["uv", "run", "python", "main.py"]
