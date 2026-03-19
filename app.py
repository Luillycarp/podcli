"""FastAPI wrapper — exposes podcli backend as REST API for HF Spaces."""
import os
import json
import traceback
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException, Security, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader

app = FastAPI(title="podcli API", version="1.4.0")

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("PODCLI_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("/tmp/podcli_output")
STATUS_DIR  = Path("/tmp/podcli_status")
BACKEND_DIR = "/app/backend"

def _ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

_ensure_dirs()

def _set_job(job_id: str, status: str):
    _ensure_dirs()
    (STATUS_DIR / f"{job_id}.json").write_text(json.dumps({"job_id": job_id, "status": status}))

def _get_job(job_id: str) -> str:
    p = STATUS_DIR / f"{job_id}.json"
    if p.exists():
        return json.loads(p.read_text()).get("status", "unknown")
    return "not_found"

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.4.0"}

# ── Transcribe (file upload) ──────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile, _=Security(verify_key)):
    suffix = Path(file.filename or "audio").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    return _run_transcription(tmp_path)

# ── Transcribe URL (Space downloads directly) ─────────────────────────────────
@app.post("/transcribe_url")
async def transcribe_url(
    video_url: str = Form(...),
    start: float = Form(0),
    end: float = Form(300),
    _=Security(verify_key),
):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        dl = subprocess.run(
            [
                "yt-dlp",
                "--download-sections", f"*{start}-{end}",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
                "-o", tmp_path,
                video_url,
            ],
            capture_output=True, text=True, timeout=300,
        )
        if dl.returncode != 0:
            raise HTTPException(500, detail={"error": "yt-dlp failed", "stderr": dl.stderr[-1000:]})
        return _run_transcription(tmp_path)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail="Download timeout")
    finally:
        # Always clean up temp file
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

def _run_transcription(tmp_path: str) -> JSONResponse:
    model_size = os.getenv("WHISPER_MODEL", "base")
    script = f"""
import json, sys
sys.path.insert(0, '{BACKEND_DIR}')
from services.transcription import transcribe_file
result = transcribe_file('{tmp_path}', model_size='{model_size}', enable_diarization=False)
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            ["python", "-c", script],
            capture_output=True, text=True, timeout=600, cwd="/app",
        )
        Path(tmp_path).unlink(missing_ok=True)
        if result.returncode != 0:
            raise HTTPException(500, detail={
                "error": "transcription failed",
                "stderr": result.stderr[-2000:],
                "stdout": result.stdout[-500:],
            })
        return JSONResponse(content=json.loads(result.stdout))
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(504, detail="Transcription timeout (>10 min)")
    except HTTPException:
        raise
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(500, detail={"error": str(e), "trace": traceback.format_exc()[-1000:]})

# ── Create Clip ───────────────────────────────────────────────────────────────
@app.post("/clip/{job_id}")
async def create_clip(
    job_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    start: float = Form(...),
    end: float = Form(...),
    caption_style: str = Form("hormozi"),
    crop: str = Form("center"),
    transcript_json: str = Form("[]"),   # JSON string: [{word, start, end}, ...]
    _=Security(verify_key),
):
    suffix = Path(file.filename or "segment").suffix or ".mp4"
    input_path = OUTPUT_DIR / f"{job_id}_input{suffix}"
    with open(input_path, "wb") as f_out:
        f_out.write(await file.read())

    # Parse transcript words if provided
    try:
        transcript_words = json.loads(transcript_json)
    except Exception:
        transcript_words = []

    _set_job(job_id, "processing")
    background_tasks.add_task(
        _render, job_id, str(input_path), start, end, caption_style, crop, transcript_words
    )
    return {"job_id": job_id, "status": "processing"}

@app.get("/clip/{job_id}")
def get_clip(job_id: str, _=Security(verify_key)):
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if output_path.exists():
        return FileResponse(str(output_path), media_type="video/mp4",
                            filename=f"clip_{job_id}.mp4")
    return {"job_id": job_id, "status": _get_job(job_id)}

@app.get("/clips")
def list_clips(_=Security(verify_key)):
    clips = [f.stem for f in OUTPUT_DIR.glob("*.mp4") if "_input" not in f.name]
    return {"clips": clips, "count": len(clips)}

# ── Background render ─────────────────────────────────────────────────────────
def _render(
    job_id: str,
    input_path: str,
    start: float,
    end: float,
    caption_style: str,
    crop: str,
    transcript_words: list,
):
    output_path = str(OUTPUT_DIR / f"{job_id}.mp4")
    script = f"""
import sys, json
sys.path.insert(0, '{BACKEND_DIR}')
from services.clip_generator import generate_clip

transcript_words = {json.dumps(transcript_words)}

result = generate_clip(
    video_path={repr(input_path)},
    start_second=0,
    end_second={end - start},
    caption_style={repr(caption_style)},
    crop_strategy={repr(crop)},
    transcript_words=transcript_words if transcript_words else None,
    title={repr(job_id)},
    output_dir={repr(str(OUTPUT_DIR))},
)
# rename to expected job_id.mp4
import os, shutil
expected = os.path.join({repr(str(OUTPUT_DIR))}, {repr(job_id)} + '.mp4')
if result['output_path'] != expected:
    shutil.move(result['output_path'], expected)
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            ["python", "-c", script],
            capture_output=True, text=True, timeout=300, cwd="/app",
        )
        if result.returncode != 0:
            _set_job(job_id, f"error:{result.stderr[-500:]}")
        else:
            _set_job(job_id, "done")
    except subprocess.TimeoutExpired:
        _set_job(job_id, "error:render timeout")
    except Exception as e:
        _set_job(job_id, f"error:{e}")
    finally:
        Path(input_path).unlink(missing_ok=True)

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
