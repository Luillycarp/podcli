"""Lightweight MCP server — runs on your laptop, delegates heavy work to HF Space.

Usage in opencode.json / claude_desktop_config.json:
  {
    \"mcpServers\": {
      \"podcast-clipper\": {
        \"command\": \"python\",
        \"args\": [\"/absolute/path/to/mcp_local.py\"]
      }
    }
  }

Required env vars (set in .env or export before running):
  HF_SPACE_URL   — e.g. https://luillycarp-podcli.hf.space
  PODCLI_API_KEY — optional, must match the Space Secret
"""

import os
import sys
import time
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
    print(
        "[podcli-mcp] WARNING: HF_SPACE_URL not set — clip rendering disabled.",
        file=sys.stderr,
    )

mcp = FastMCP("podcast-clipper")


# ─── Tool 1: get_transcript ───────────────────────────────────────────────────
@mcp.tool()
def get_transcript(video_id: str, languages: list[str] = ["es", "en"]) -> dict:
    """
    Fetch transcript with timestamps from a YouTube video.
    Tries native captions first (instant); falls back to HF Space Whisper if unavailable.

    Args:
        video_id:  YouTube video ID (the part after ?v=), e.g. '5mAUPzli74Y'
        languages: Preferred languages in order of preference.

    Returns:
        {"source": "youtube"|"whisper", "segments": [{text, start, duration}, ...]}
    """
    try:
        yt = YouTubeTranscriptApi()
        transcript_list = yt.list(video_id)

        # Try to find a transcript in preferred languages
        transcript = None
        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except:
                continue

        if not transcript:
            # Try auto-generated
            try:
                transcript = transcript_list.find_transcript(["en"])
            except:
                pass

        if transcript:
            fetched = transcript.fetch()
            # Convert to segments format
            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in fetched
            ]
            return {"source": "youtube", "segments": segments}

        return {"source": "unavailable", "segments": [], "hint": "No captions found"}
        return {"source": "youtube", "segments": segments}
    except (TranscriptsDisabled, NoTranscriptFound):
        return {
            "source": "unavailable",
            "segments": [],
            "hint": (
                f"No captions found for {video_id}. "
                "Download the audio and call transcribe_audio() instead."
            ),
        }


# ─── Tool 2: transcribe_audio ─────────────────────────────────────────────────
@mcp.tool()
def transcribe_audio(video_url: str, start: float = 0, end: float = 300) -> dict:
    """
    Download a segment and transcribe it via HF Space Whisper.
    Use only when get_transcript() returns source='unavailable'.

    Args:
        video_url: Full YouTube URL.
        start:     Start time in seconds.
        end:       End time in seconds.
    """
    if not HF_SPACE:
        return {"error": "HF_SPACE_URL not configured"}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(
        [
            "yt-dlp",
            "--download-sections",
            f"*{start}-{end}",
            "--js-runtimes",
            "node",
            "-o",
            tmp_path,
            video_url,
        ],
        check=True,
    )

    with open(tmp_path, "rb") as f:
        r = httpx.post(
            f"{HF_SPACE}/transcribe",
            files={"file": ("segment.mp4", f, "video/mp4")},
            headers=HEADERS,
            timeout=600,
        )
    Path(tmp_path).unlink(missing_ok=True)
    r.raise_for_status()
    return {"source": "whisper", "segments": r.json()}


# ─── Tool 3: create_clip ──────────────────────────────────────────────────────
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
    Download a video segment and send it to the HF Space for rendering.
    Returns a job_id — poll get_clip_status() until done.

    Args:
        video_url:     Full YouTube URL.
        start:         Clip start time in seconds.
        end:           Clip end time in seconds.
        caption_style: 'hormozi' | 'branded' | 'karaoke' | 'subtle'
        crop:          'center' | 'face'
        output_name:   Optional human-readable label for the clip.
    """
    if not HF_SPACE:
        return {"error": "HF_SPACE_URL not configured"}

    import uuid

    job_id = output_name.replace(" ", "_") or str(uuid.uuid4())[:8]

    # Download only the needed segment
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(
        [
            "yt-dlp",
            "--download-sections",
            f"*{start}-{end}",
            "--js-runtimes",
            "node",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "-o",
            tmp_path,
            video_url,
        ],
        check=True,
    )

    with open(tmp_path, "rb") as f:
        r = httpx.post(
            f"{HF_SPACE}/clip/{job_id}",
            params={
                "start": start,
                "end": end,
                "caption_style": caption_style,
                "crop": crop,
            },
            files={"file": ("segment.mp4", f, "video/mp4")},
            headers=HEADERS,
            timeout=120,
        )
    Path(tmp_path).unlink(missing_ok=True)
    r.raise_for_status()
    return r.json()


# ─── Tool 4: get_clip_status ──────────────────────────────────────────────────
@mcp.tool()
def get_clip_status(job_id: str, download: bool = True) -> dict:
    """
    Check the render status of a clip. If done and download=True, saves it locally.

    Args:
        job_id:   The job_id returned by create_clip().
        download: If True and clip is ready, download to ./clips/
    """
    if not HF_SPACE:
        return {"error": "HF_SPACE_URL not configured"}

    r = httpx.get(f"{HF_SPACE}/clip/{job_id}", headers=HEADERS, timeout=30)

    if r.headers.get("content-type", "").startswith("video/mp4"):
        if download:
            clips_dir = Path("./clips")
            clips_dir.mkdir(exist_ok=True)
            out = clips_dir / f"{job_id}.mp4"
            out.write_bytes(r.content)
            return {
                "status": "done",
                "local_path": str(out),
                "size_mb": round(len(r.content) / 1_000_000, 2),
            }
        return {"status": "done", "bytes": len(r.content)}

    return r.json()


# ─── Tool 5: batch_create_clips ───────────────────────────────────────────────
@mcp.tool()
def batch_create_clips(
    video_url: str,
    clips: list[dict],
    caption_style: str = "hormozi",
    crop: str = "center",
) -> list[dict]:
    """
    Submit multiple clips at once and wait for all to finish.

    Args:
        video_url:     Full YouTube URL.
        clips:         List of {start, end, name} dicts.
        caption_style: Caption style for all clips.
        crop:          Crop strategy for all clips.

    Example clips list:
        [{"start": 120, "end": 180, "name": "regla_1"},
         {"start": 540, "end": 600, "name": "regla_2"}]
    """
    results = []
    job_ids = []

    # Submit all
    for clip in clips:
        r = create_clip(
            video_url=video_url,
            start=clip["start"],
            end=clip["end"],
            caption_style=caption_style,
            crop=crop,
            output_name=clip.get("name", ""),
        )
        job_ids.append(r.get("job_id", ""))
        results.append(
            {"clip": clip.get("name"), "job_id": r.get("job_id"), "submitted": True}
        )

    # Poll until all done (max 10 min)
    deadline = time.time() + 600
    pending = set(job_ids)
    while pending and time.time() < deadline:
        time.sleep(8)
        for jid in list(pending):
            s = get_clip_status(jid, download=True)
            if s.get("status") in ("done",) or "local_path" in s:
                pending.discard(jid)
                for r in results:
                    if r["job_id"] == jid:
                        r["local_path"] = s.get("local_path", "")
                        r["status"] = "done"
            elif str(s.get("status", "")).startswith("error"):
                pending.discard(jid)
                for r in results:
                    if r["job_id"] == jid:
                        r["status"] = s["status"]

    return results


# ─── Tool 6: space_health ─────────────────────────────────────────────────────
@mcp.tool()
def space_health() -> dict:
    """Ping the HF Space to check if it's awake."""
    if not HF_SPACE:
        return {"error": "HF_SPACE_URL not configured"}
    try:
        r = httpx.get(f"{HF_SPACE}/health", timeout=15, headers=HEADERS)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {
            "status": "error",
            "message": "Unexpected response",
            "text": r.text[:200],
        }
    except httpx.TimeoutException:
        return {"status": "sleeping", "hint": "Space is waking up, retry in 30s"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
