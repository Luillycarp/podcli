"""Lightweight MCP server — runs on your laptop, delegates heavy work to HF Space.

Usage in opencode.json / config.json:
  {
    \"mcpServers\": {
      \"podcli-hf\": {
        \"command\": \"python\",
        \"args\": [\"/absolute/path/to/mcp_local.py\"],
        \"env\": {
          \"HF_SPACE_URL\": \"https://luilly3-podcli.hf.space\",
          \"PODCLI_API_KEY\": \"...\"
        }
      }
    }
  }
"""

import os
import sys
import json
import time
import threading
import subprocess
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from mcp.server.fastmcp import FastMCP

load_dotenv()

HF_SPACE = os.getenv("HF_SPACE_URL", "").rstrip("/")
API_KEY = os.getenv("PODCLI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

HEADERS = {}
if API_KEY:
    HEADERS["X-API-Key"] = API_KEY
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

if not HF_SPACE:
    print("[podcli-mcp] WARNING: HF_SPACE_URL not set", file=sys.stderr)

# Local status store — survives MCP restarts (not Space restarts)
STATUS_DIR = Path(tempfile.gettempdir()) / "podcli_jobs"
STATUS_DIR.mkdir(exist_ok=True)

mcp = FastMCP("podcast-clipper")


# ── Helpers ───────────────────────────────────────────────────────────────────────────
def _set_status(job_id: str, data: dict):
    (STATUS_DIR / f"{job_id}.json").write_text(json.dumps(data))

def _get_status(job_id: str) -> dict:
    p = STATUS_DIR / f"{job_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"status": "not_found"}

def _wake_space(retries: int = 3) -> bool:
    """Ping Space; wait up to 60s if sleeping. Returns True if alive."""
    if not HF_SPACE:
        return False
    for i in range(retries):
        try:
            r = httpx.get(f"{HF_SPACE}/health", timeout=20, headers=HEADERS)
            if r.status_code == 200:
                return True
        except httpx.TimeoutException:
            pass
        if i < retries - 1:
            time.sleep(20)
    return False

def _yt_download(url: str, start: float, end: float, out_path: str):
    """Download a YT segment to out_path."""
    subprocess.run(
        [
            "yt-dlp",
            "--download-sections", f"*{start}-{end}",
            "--js-runtimes", "node",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "-o", out_path,
            url,
        ],
        check=True,
        capture_output=True,
    )


# ── Tool 1: get_transcript ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_transcript(video_id: str, languages: list[str] = ["es", "en"]) -> dict:
    """
    Fetch transcript with timestamps from a YouTube video.
    Tries native captions first (instant); returns hint to call transcribe_audio if unavailable.

    Args:
        video_id:  YouTube video ID (e.g. '5mAUPzli74Y')
        languages: Preferred languages in order.
    """
    # Check disk cache (24h)
    cache_key = f"transcript_{video_id}_{'-'.join(languages)}"
    cache_file = STATUS_DIR / f"{cache_key}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400:
            return json.loads(cache_file.read_text())

    try:
        yt = YouTubeTranscriptApi()
        transcript_list = yt.list(video_id)

        transcript = None
        # Try preferred languages
        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except Exception:
                continue

        # FIX: also try auto-generated in any available language
        if transcript is None:
            try:
                available = list(transcript_list)
                if available:
                    transcript = available[0]  # take whatever is available
            except Exception:
                pass

        if transcript:
            fetched = transcript.fetch()
            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in fetched
            ]
            result = {"source": "youtube", "language": transcript.language_code, "segments": segments}
            cache_file.write_text(json.dumps(result))
            return result

        result = {
            "source": "unavailable",
            "segments": [],
            "hint": f"No captions for {video_id}. Call transcribe_audio(video_url, start, end) instead.",
        }
        return result

    except (TranscriptsDisabled, NoTranscriptFound):
        return {
            "source": "unavailable",
            "segments": [],
            "hint": f"Captions disabled for {video_id}. Call transcribe_audio(video_url, start, end) instead.",
        }


# ── Tool 2: transcribe_audio (ASYNC) ────────────────────────────────────────────────────────────
@mcp.tool()
def transcribe_audio(video_url: str, start: float = 0, end: float = 300, job_id: str = "") -> dict:
    """
    Download a segment and transcribe via HF Space Whisper. ASYNC — returns immediately.
    Poll get_clip_status(job_id) until status='done'.

    Args:
        video_url: Full YouTube URL.
        start:     Start time in seconds.
        end:       End time in seconds.
        job_id:    Optional custom ID; auto-generated if empty.
    """
    if not HF_SPACE:
        return {"error": "HF_SPACE_URL not configured"}

    import uuid
    jid = job_id or f"tr_{str(uuid.uuid4())[:8]}"
    _set_status(jid, {"status": "downloading", "type": "transcription", "started": time.time()})

    def _worker():
        try:
            _wake_space()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            _yt_download(video_url, start, end, tmp_path)
            _set_status(jid, {"status": "uploading", "type": "transcription"})
            with open(tmp_path, "rb") as f:
                r = httpx.post(
                    f"{HF_SPACE}/transcribe",
                    files={"file": ("segment.mp4", f, "video/mp4")},
                    headers=HEADERS,
                    timeout=600,
                )
            Path(tmp_path).unlink(missing_ok=True)
            r.raise_for_status()
            _set_status(jid, {"status": "done", "type": "transcription", "result": r.json()})
        except Exception as e:
            _set_status(jid, {"status": f"error:{e}", "type": "transcription"})

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": jid, "status": "downloading", "hint": "Poll get_clip_status(job_id) for result"}


# ── Tool 3: create_clip (ASYNC) ──────────────────────────────────────────────────────────────────
@mcp.tool()
def create_clip(
    video_url: str,
    start: float,
    end: float,
    caption_style: str = "hormozi",
    crop: str = "center",
    output_name: str = "",
) -> dict:
    """
    Download a segment and send it to HF Space for rendering. ASYNC — returns immediately.
    Poll get_clip_status(job_id) every 15-30s until status='done'.

    Args:
        video_url:     Full YouTube URL.
        start:         Clip start in seconds.
        end:           Clip end in seconds.
        caption_style: 'hormozi' | 'branded' | 'karaoke' | 'subtle'
        crop:          'center' | 'face'
        output_name:   Human-readable label (used as job_id).
    """
    if not HF_SPACE:
        return {"error": "HF_SPACE_URL not configured"}

    import uuid
    job_id = output_name.replace(" ", "_") or str(uuid.uuid4())[:8]
    _set_status(job_id, {
        "status": "downloading",
        "url": video_url, "start": start, "end": end,
        "caption_style": caption_style, "crop": crop,
        "started": time.time()
    })

    def _worker():
        try:
            # 1. Wake Space before downloading (parallel)
            wake_t = threading.Thread(target=_wake_space, daemon=True)
            wake_t.start()

            # 2. Download
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            _yt_download(video_url, start, end, tmp_path)

            # 3. Wait for Space to be awake
            wake_t.join(timeout=60)
            _set_status(job_id, {"status": "uploading"})

            # 4. Upload as multipart/form-data (Form fields + file)
            with open(tmp_path, "rb") as f:
                r = httpx.post(
                    f"{HF_SPACE}/clip/{job_id}",
                    data={
                        "start": str(start),
                        "end": str(end),
                        "caption_style": caption_style,
                        "crop": crop,
                    },
                    files={"file": ("segment.mp4", f, "video/mp4")},
                    headers=HEADERS,
                    timeout=120,
                )
            Path(tmp_path).unlink(missing_ok=True)
            r.raise_for_status()
            space_response = r.json()
            _set_status(job_id, {"status": "rendering", "space_job": space_response})

            # 5. Poll Space until render done
            deadline = time.time() + 600
            while time.time() < deadline:
                time.sleep(10)
                sr = httpx.get(f"{HF_SPACE}/clip/{job_id}", headers=HEADERS, timeout=30)
                if sr.headers.get("content-type", "").startswith("video/mp4"):
                    # Download to local clips dir
                    clips_dir = Path("./clips")
                    clips_dir.mkdir(exist_ok=True)
                    out = clips_dir / f"{job_id}.mp4"
                    out.write_bytes(sr.content)
                    _set_status(job_id, {
                        "status": "done",
                        "local_path": str(out.resolve()),
                        "size_mb": round(len(sr.content) / 1_000_000, 2),
                    })
                    return
                sj = sr.json()
                if str(sj.get("status", "")).startswith("error"):
                    _set_status(job_id, {"status": sj["status"]})
                    return

            _set_status(job_id, {"status": "error:render timeout"})
        except Exception as e:
            _set_status(job_id, {"status": f"error:{e}"})

    threading.Thread(target=_worker, daemon=True).start()
    return {
        "job_id": job_id,
        "status": "downloading",
        "hint": "Call get_clip_status(job_id) in ~2 min to check progress",
    }


# ── Tool 4: get_clip_status ───────────────────────────────────────────────────────────────────
@mcp.tool()
def get_clip_status(job_id: str) -> dict:
    """
    Check the status of a create_clip or transcribe_audio job.

    Returns:
        {status: 'downloading'|'uploading'|'rendering'|'done'|'error:...'|'not_found',
         local_path?: str,   # when done (clips only)
         result?: dict}      # when done (transcriptions only)
    """
    return _get_status(job_id)


# ── Tool 5: batch_create_clips ─────────────────────────────────────────────────────────────────
@mcp.tool()
def batch_create_clips(
    video_url: str,
    clips: list[dict],
    caption_style: str = "hormozi",
    crop: str = "center",
) -> list[dict]:
    """
    Submit multiple clips IN PARALLEL and return immediately with job_ids.
    Each clip is processed concurrently — no sequential waiting.
    Call get_clip_status(job_id) on each to track progress.

    Args:
        video_url:     Full YouTube URL.
        clips:         [{start, end, name}, ...]
        caption_style: Applied to all clips.
        crop:          Applied to all clips.

    Example:
        [{"start": 120, "end": 180, "name": "hook_1"},
         {"start": 540, "end": 600, "name": "cta_end"}]
    """
    results = []
    # Submit all in parallel (each create_clip spawns its own thread)
    for clip in clips:
        r = create_clip(
            video_url=video_url,
            start=clip["start"],
            end=clip["end"],
            caption_style=caption_style,
            crop=crop,
            output_name=clip.get("name", ""),
        )
        results.append({"name": clip.get("name"), "job_id": r["job_id"], "status": "submitted"})

    return {
        "submitted": len(results),
        "jobs": results,
        "hint": "All downloads running in parallel. Poll get_clip_status(job_id) per clip."
    }


# ── Tool 6: space_health ──────────────────────────────────────────────────────────────────────────
@mcp.tool()
def space_health() -> dict:
    """Ping the HF Space. Returns status and whether it needed to wake up."""
    if not HF_SPACE:
        return {"error": "HF_SPACE_URL not configured"}
    t0 = time.time()
    alive = _wake_space(retries=1)
    return {
        "status": "ok" if alive else "sleeping",
        "response_ms": round((time.time() - t0) * 1000),
        "space_url": HF_SPACE,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
