"""FastAPI wrapper — exposes podcli backend as REST API for HF Spaces."""

import os
import traceback
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException, Security, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

app = FastAPI(title="podcli API", version="1.0.0")

# ── Auth ──────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("PODCLI_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ── Paths ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/tmp/podcli_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, str] = {}


# ── Health ───────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Transcribe ────────────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile, _=Security(verify_key)):
    """
    Upload a video/audio file, returns word-level transcript.
    [{word, start, end, confidence, speaker}, ...]
    """
    suffix = Path(file.filename or "audio").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # FIX: correct kwarg is model_size= not model=
    model_size = os.getenv("WHISPER_MODEL", "base")
    script = f"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
from services.transcription import transcribe_file
result = transcribe_file('{tmp_path}', model_size='{model_size}', enable_diarization=False)
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            ["python", "-c", script],
            capture_output=True, text=True, timeout=600,
            cwd="/app"
        )
        Path(tmp_path).unlink(missing_ok=True)

        if result.returncode != 0:
            # Return full stderr so we can debug from Space logs
            raise HTTPException(500, detail={
                "error": "transcription failed",
                "stderr": result.stderr[-2000:],
                "stdout": result.stdout[-500:]
            })

        return JSONResponse(content=json.loads(result.stdout))

    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(504, detail="Transcription timeout (>10 min)")
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(500, detail={"error": str(e), "trace": traceback.format_exc()[-1000:]})


# ── Create Clip ───────────────────────────────────────────────────────────────────
# FIX: accept params as query OR form fields — multipart requests can't mix JSON body + file
@app.post("/clip/{job_id}")
async def create_clip(
    job_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    start: float = Form(...),
    end: float = Form(...),
    caption_style: str = Form("hormozi"),
    crop: str = Form("center"),
    _=Security(verify_key),
):
    """
    Upload pre-cut video segment as multipart/form-data.
    Fields: start, end, caption_style, crop + file.
    Returns {job_id, status: 'processing'} — poll GET /clip/{job_id}.
    """
    suffix = Path(file.filename or "segment").suffix or ".mp4"
    input_path = OUTPUT_DIR / f"{job_id}_input{suffix}"
    with open(input_path, "wb") as f_out:
        f_out.write(await file.read())

    JOBS[job_id] = "processing"
    background_tasks.add_task(
        _render, job_id, str(input_path),
        start, end, caption_style, crop
    )
    return {"job_id": job_id, "status": "processing"}


@app.get("/clip/{job_id}")
def get_clip(job_id: str, _=Security(verify_key)):
    """Returns MP4 when ready, or {job_id, status} JSON."""
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if output_path.exists():
        return FileResponse(str(output_path), media_type="video/mp4",
                            filename=f"clip_{job_id}.mp4")
    return {"job_id": job_id, "status": JOBS.get(job_id, "not_found")}


@app.get("/clips")
def list_clips(_=Security(verify_key)):
    clips = [f.name for f in OUTPUT_DIR.glob("*.mp4") if "_input" not in f.name]
    return {"clips": clips, "count": len(clips)}


# ── Background render ────────────────────────────────────────────────────────────────────
def _render(job_id: str, input_path: str,
           start: float, end: float,
           caption_style: str, crop: str):
    output_path = str(OUTPUT_DIR / f"{job_id}.mp4")
    duration = end - start
    script = f"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
from services.video_processor import process_clip
process_clip(
    input_path='{input_path}',
    output_path='{output_path}',
    start=0,
    end={duration},
    caption_style='{caption_style}',
    crop='{crop}',
)
"""
    try:
        result = subprocess.run(
            ["python", "-c", script],
            capture_output=True, text=True, timeout=300,
            cwd="/app"
        )
        if result.returncode != 0:
            JOBS[job_id] = f"error:{result.stderr[-300:]}"
        else:
            JOBS[job_id] = "done"
    except Exception as e:
        JOBS[job_id] = f"error:{e}"
    finally:
        Path(input_path).unlink(missing_ok=True)


# ── Entrypoint ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
