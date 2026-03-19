"""FastAPI wrapper — exposes podcli backend as REST API for HF Spaces."""
import os
import uuid
import json
import asyncio
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException, Security
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

app = FastAPI(title="podcli API", version="1.0.0")

# ── Auth (optional) ──────────────────────────────────────────────────────────
API_KEY = os.getenv("PODCLI_API_KEY", "")  # set in HF Space Secrets; empty = no auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/tmp/podcli_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, str] = {}  # job_id → "processing" | "done" | "error:<msg>"


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Transcribe ────────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile, _=Security(verify_key)):
    """
    Upload a video/audio file, get back a word-level transcript.
    Returns JSON list: [{word, start, end, speaker?}, ...]
    """
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    model = os.getenv("WHISPER_MODEL", "base")
    result = subprocess.run(
        ["python", "-c", f"""
import json, sys
sys.path.insert(0, 'backend')
from services.transcription import transcribe_file
result = transcribe_file('{tmp_path}', model='{model}')
print(json.dumps(result))
"""],
        capture_output=True, text=True, timeout=600
    )
    os.unlink(tmp_path)

    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr[-500:])

    return JSONResponse(content=json.loads(result.stdout))


# ── Create Clip ───────────────────────────────────────────────────────────────
class ClipRequest(BaseModel):
    start: float                          # seconds
    end: float                            # seconds
    caption_style: str = "hormozi"        # hormozi | branded | karaoke | subtle
    crop: str = "center"                  # center | face
    logo_path: str | None = None


@app.post("/clip/{job_id}")
async def create_clip(
    job_id: str,
    payload: ClipRequest,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    _=Security(verify_key),
):
    """
    Upload the pre-cut segment + render params.
    Returns immediately with job_id; poll GET /clip/{job_id} for the result.
    """
    suffix = Path(file.filename).suffix or ".mp4"
    input_path = OUTPUT_DIR / f"{job_id}_input{suffix}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    JOBS[job_id] = "processing"
    background_tasks.add_task(_render, job_id, str(input_path), payload)
    return {"job_id": job_id, "status": "processing"}


@app.get("/clip/{job_id}")
def get_clip(job_id: str, _=Security(verify_key)):
    """Returns the rendered MP4 when ready, or status JSON."""
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if output_path.exists():
        return FileResponse(str(output_path), media_type="video/mp4",
                            filename=f"clip_{job_id}.mp4")
    status = JOBS.get(job_id, "not_found")
    return {"job_id": job_id, "status": status}


@app.get("/clips")
def list_clips(_=Security(verify_key)):
    """List all rendered clips."""
    clips = [f.name for f in OUTPUT_DIR.glob("*.mp4") if "_input" not in f.name]
    return {"clips": clips}


# ── Background render task ────────────────────────────────────────────────────
def _render(job_id: str, input_path: str, p: ClipRequest):
    output_path = str(OUTPUT_DIR / f"{job_id}.mp4")
    try:
        cmd = [
            "python", "-c", f"""
import sys
sys.path.insert(0, 'backend')
from services.video_processor import process_clip
process_clip(
    input_path='{input_path}',
    output_path='{output_path}',
    start=0,
    end={p.end - p.start},
    caption_style='{p.caption_style}',
    crop='{p.crop}',
)
"""
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            JOBS[job_id] = f"error:{result.stderr[-200:]}"
        else:
            JOBS[job_id] = "done"
    except Exception as e:
        JOBS[job_id] = f"error:{e}"
    finally:
        Path(input_path).unlink(missing_ok=True)


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
