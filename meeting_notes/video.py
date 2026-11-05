import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from .llm import LLMService

logger = logging.getLogger(__name__)


class VideoService:
    """Video frame extraction, metadata, and chunked transcription."""

    DEFAULT_CHUNK_OVERLAP_SECONDS = 10
    DEFAULT_TRANSCRIBE_MAX_WORKERS = 4

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    @classmethod
    def _transcribe_max_workers(cls) -> int:
        raw = os.getenv("LLM_TRANSCRIBE_MAX_WORKERS", "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return cls.DEFAULT_TRANSCRIBE_MAX_WORKERS

    @staticmethod
    def format_duration(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def extract_frames(
        self,
        video_path: str,
        interval_seconds: int = 30,
        max_frames: int = 8,
    ) -> list[dict]:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_f / fps

        timestamps = []
        t = interval_seconds
        while t < duration and len(timestamps) < max_frames:
            timestamps.append(min(t, duration - 0.5))
            t += interval_seconds
        if not timestamps:
            timestamps = [duration / 2]

        frames = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=72)
            b64 = base64.b64encode(buf.getvalue()).decode()
            frames.append(
                {
                    "time_seconds": ts,
                    "time_str": self.format_duration(ts),
                    "pil_image": img,
                    "base64_jpeg": b64,
                }
            )

        cap.release()
        return frames

    @staticmethod
    def _extract_keywords(text: str, limit: int = 12) -> list[str]:
        stop_words = {
            "this", "that", "with", "from", "have", "will", "your", "about", "there",
            "their", "into", "after", "before", "would", "could", "should", "were",
            "been", "being", "also", "only", "when", "what", "where", "which",
            "while", "than", "then", "them", "they", "over", "under", "more", "most",
        }
        tokens = re.findall(r"[a-zA-Z0-9]{4,}", text.lower())
        filtered = [t for t in tokens if t not in stop_words]
        if not filtered:
            return []
        return [word for word, _ in Counter(filtered).most_common(limit)]

    def enrich_frames_with_ocr(self, frames: list[dict]) -> dict:
        """Optionally run OCR on extracted frames to improve screenshot/topic alignment.

        The method is best-effort: it never raises and returns availability details.
        """
        try:
            import pytesseract
        except Exception as e:
            logger.info("pytesseract is not importable: %s", e)
            for frame in frames:
                frame["ocr_text"] = ""
                frame["ocr_keywords"] = []
            return {"available": False, "engine": "none", "reason": "pytesseract not installed"}

        try:
            version = pytesseract.get_tesseract_version()
            logger.info("pytesseract runtime available (version: %s)", version)
        except Exception as e:
            logger.warning("tesseract runtime unavailable for pytesseract fallback: %s: %s", type(e).__name__, e)
            for frame in frames:
                frame["ocr_text"] = ""
                frame["ocr_keywords"] = []
            return {
                "available": False,
                "engine": "none",
                "reason": f"tesseract runtime unavailable: {type(e).__name__}",
            }

        extracted_count = 0
        for i, frame in enumerate(frames, start=1):
            text = ""
            try:
                text = pytesseract.image_to_string(frame["pil_image"]) or ""
            except Exception as e:
                logger.warning("pytesseract failed on frame %d/%d (%s): %s", i, len(frames), frame.get("time_str", "N/A"), e)
                text = ""

            normalized = " ".join(text.split())
            if len(normalized) > 240:
                normalized = normalized[:240] + "..."

            frame["ocr_text"] = normalized
            frame["ocr_keywords"] = self._extract_keywords(normalized)
            if normalized:
                extracted_count += 1
                logger.debug("pytesseract frame %d/%d (%s) extracted: %s", i, len(frames), frame.get("time_str", "N/A"), normalized[:60])

        logger.info("pytesseract OCR completed: %d/%d frame(s) with text.", extracted_count, len(frames))
        return {"available": True, "engine": "pytesseract"}

    def get_video_info(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_f / fps
        cap.release()
        size_mb = Path(video_path).stat().st_size / 1_048_576
        return {
            "duration": duration,
            "duration_str": self.format_duration(duration),
            "fps": fps,
            "resolution": f"{width}x{height}",
            "size_mb": round(size_mb, 1),
            "filename": Path(video_path).name,
        }

    @staticmethod
    def resolve_ffmpeg_binary() -> str:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            return ffmpeg_bin

        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise RuntimeError(
                "ffmpeg is unavailable. Install ffmpeg or pip install imageio-ffmpeg."
            ) from e

    def transcribe_video_chunked(
        self,
        video_path: str,
        *,
        base_url: str,
        api_key: str,
        transcribe_model: str,
        chunk_seconds: int = 90,
        chunk_overlap_seconds: int = DEFAULT_CHUNK_OVERLAP_SECONDS,
    ) -> dict:
        info = self.get_video_info(video_path)
        duration_seconds = float(info["duration"])
        chunk_windows = self._plan_transcript_chunk_windows(
            duration_seconds=duration_seconds,
            chunk_seconds=chunk_seconds,
            overlap_seconds=chunk_overlap_seconds,
        )
        expected_chunks = len(chunk_windows)

        video = Path(video_path)
        artifacts_dir = Path.cwd() / "artifacts"
        transcripts_dir = artifacts_dir / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = transcripts_dir / f"{video.stem}.transcript.txt"
        transcript_plain_path = transcripts_dir / f"{video.stem}.full.txt"
        manifest_path = transcripts_dir / f"{video.stem}.manifest.json"
        chunk_text_dir = transcripts_dir / f"{video.stem}_chunks"
        chunk_text_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg_bin = self.resolve_ffmpeg_binary()

        client = self.llm_service.build_client(base_url, api_key)

        transcript_parts: list[str] = []
        generated_chunk_files: list[str] = []

        manifest = {
            "video_file": video.name,
            "generated_at": datetime.now().isoformat(),
            "model": transcribe_model,
            "chunk_seconds": chunk_seconds,
            "chunk_overlap_seconds": chunk_overlap_seconds,
            "chunk_step_seconds": chunk_seconds - chunk_overlap_seconds,
            "duration_seconds": round(duration_seconds, 3),
            "expected_chunks": expected_chunks,
            "actual_chunks": 0,
            "coverage_ratio": 0.0,
            "status": "started",
        }

        try:
            with tempfile.TemporaryDirectory() as td:
                chunk_window_manifest: list[dict] = []

                def _process_chunk(i: int, window: dict) -> tuple[int, str, dict]:
                    chunk = Path(td) / f"audio_chunk_{i:03d}.mp3"
                    chunk_duration = max(0.1, window["end_seconds"] - window["start_seconds"])
                    split_cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-ss",
                        f"{window['start_seconds']:.3f}",
                        "-i",
                        str(video),
                        "-t",
                        f"{chunk_duration:.3f}",
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-b:a",
                        "32k",
                        str(chunk),
                    ]
                    result = subprocess.run(split_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        stderr_tail = result.stderr[-800:] if result.stderr else "unknown error"
                        raise RuntimeError(
                            f"ffmpeg split failed for chunk {i} "
                            f"({window['start_seconds']:.3f}s-{window['end_seconds']:.3f}s): {stderr_tail}"
                        )

                    if not chunk.exists() or chunk.stat().st_size == 0:
                        raise RuntimeError(f"No audio content was created for chunk {i}.")

                    with open(chunk, "rb") as af:
                        tr = client.audio.transcriptions.create(
                            model=transcribe_model,
                            file=af,
                        )
                    text_part = tr if isinstance(tr, str) else getattr(tr, "text", str(tr))
                    return i, text_part, window

                max_workers = min(self._transcribe_max_workers(), max(1, len(chunk_windows)))
                raw_results: dict[int, tuple[str, dict]] = {}
                if max_workers <= 1 or len(chunk_windows) <= 1:
                    for i, window in enumerate(chunk_windows, start=1):
                        _, text_part, win = _process_chunk(i, window)
                        raw_results[i] = (text_part, win)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(copy_context().run, _process_chunk, i, window): i
                            for i, window in enumerate(chunk_windows, start=1)
                        }
                        for future in as_completed(futures):
                            i, text_part, win = future.result()
                            raw_results[i] = (text_part, win)

                previous_text_part = ""
                for i in sorted(raw_results):
                    text_part, window = raw_results[i]
                    merged_text_part = text_part
                    if previous_text_part:
                        merged_text_part = self._trim_overlapping_prefix(previous_text_part, text_part)
                        if not merged_text_part.strip():
                            merged_text_part = text_part
                    transcript_parts.append(f"[Chunk {i}]\n{merged_text_part}\n")
                    previous_text_part = text_part

                    chunk_text_path = chunk_text_dir / f"chunk_{i:03d}.txt"
                    chunk_text_path.write_text(text_part, encoding="utf-8")
                    generated_chunk_files.append(str(chunk_text_path))
                    chunk_window_manifest.append(
                        {
                            "chunk_index": i,
                            "start_seconds": round(window["start_seconds"], 3),
                            "end_seconds": round(window["end_seconds"], 3),
                        }
                    )

            full_transcript = "\n".join(transcript_parts).strip()
            if not full_transcript:
                raise RuntimeError("Transcription returned empty text.")

            transcript_header = (
                f"# Raw Transcript\n"
                f"video_file: {video.name}\n"
                f"generated_at: {datetime.now().isoformat()}\n"
                f"model: {transcribe_model}\n"
                f"chunk_seconds: {chunk_seconds}\n"
                f"chunks: {len(transcript_parts)}\n\n"
            )
            transcript_path.write_text(transcript_header + full_transcript, encoding="utf-8")
            transcript_plain_path.write_text(full_transcript, encoding="utf-8")

            manifest["actual_chunks"] = len(transcript_parts)
            manifest["coverage_ratio"] = (
                round(len(transcript_parts) / expected_chunks, 3) if expected_chunks else 1.0
            )
            manifest["status"] = "ok"
            manifest["chunk_text_files"] = generated_chunk_files
            manifest["chunk_windows"] = chunk_window_manifest
            manifest["transcript_file"] = str(transcript_path)
            manifest["plain_transcript_file"] = str(transcript_plain_path)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            return {
                "transcript_text": transcript_path.read_text(encoding="utf-8"),
                "transcript_path": str(transcript_path),
                "transcript_plain_path": str(transcript_plain_path),
                "manifest_path": str(manifest_path),
                "manifest": manifest,
            }
        except Exception:
            if transcript_path.exists():
                fallback_manifest = (
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists()
                    else manifest
                )
                return {
                    "transcript_text": transcript_path.read_text(encoding="utf-8"),
                    "transcript_path": str(transcript_path),
                    "transcript_plain_path": str(transcript_plain_path),
                    "manifest_path": str(manifest_path),
                    "manifest": fallback_manifest,
                }
            raise

    @staticmethod
    def _plan_transcript_chunk_windows(
        *,
        duration_seconds: float,
        chunk_seconds: int,
        overlap_seconds: int,
    ) -> list[dict]:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds cannot be negative.")
        if overlap_seconds >= chunk_seconds:
            raise ValueError("overlap_seconds must be smaller than chunk_seconds.")

        if duration_seconds <= 0:
            return [{"start_seconds": 0.0, "end_seconds": 0.0}]

        step_seconds = float(chunk_seconds - overlap_seconds)
        windows: list[dict] = []
        start = 0.0
        duration = float(duration_seconds)
        chunk_size = float(chunk_seconds)
        while start < duration:
            end = min(start + chunk_size, duration)
            windows.append({"start_seconds": start, "end_seconds": end})
            if end >= duration:
                break
            start += step_seconds
        return windows

    @staticmethod
    def _trim_overlapping_prefix(
        previous_text: str,
        current_text: str,
        *,
        min_overlap_tokens: int = 5,
        max_overlap_tokens: int = 80,
    ) -> str:
        prev = (previous_text or "").strip()
        curr = (current_text or "").strip()
        if not prev or not curr:
            return current_text

        token_pattern = re.compile(r"\b\w+\b")
        prev_tokens = [m.group(0).lower() for m in token_pattern.finditer(prev)]
        curr_matches = list(token_pattern.finditer(curr))
        curr_tokens = [m.group(0).lower() for m in curr_matches]
        if not prev_tokens or not curr_tokens:
            return current_text

        max_k = min(len(prev_tokens), len(curr_tokens), max_overlap_tokens)
        for k in range(max_k, min_overlap_tokens - 1, -1):
            if prev_tokens[-k:] == curr_tokens[:k]:
                trim_at = curr_matches[k - 1].end()
                return curr[trim_at:].lstrip(" \t\r\n.,;:-")
        return current_text
