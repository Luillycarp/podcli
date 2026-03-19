"""
Transcription service using faster-whisper + optional speaker diarization.

faster-whisper is a CTranslate2-based reimplementation of Whisper:
- Same model weights / identical transcription quality
- ~4x faster on CPU, lower RAM footprint (no PyTorch required)
- Supports int8 quantization for HF Spaces free tier

Produces word-level timestamps with speaker labels by:
1. Running faster-whisper for speech-to-text with word timing
2. Running pyannote speaker diarization (if available)
3. Merging speaker labels onto each word and segment
"""

import os
import tempfile
from typing import Optional, Callable


def transcribe_file(
    file_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    enable_diarization: bool = True,
    num_speakers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Transcribe a video/audio file with word-level timestamps and speaker detection.

    Returns:
        {
            "transcript": str,
            "segments": [{id, start, end, text, speaker}, ...],
            "words": [{word, start, end, confidence, speaker}, ...],
            "duration": float,
            "language": str,
            "speakers": {num_speakers, speakers: {SPEAKER_00: {total_time, segments, label}, ...}},
            "speaker_segments": [{speaker, start, end}, ...]
        }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # ================================================================
    # Step 1: faster-whisper transcription
    # ================================================================
    if progress_callback:
        progress_callback(5, "Loading faster-whisper model...")

    from faster_whisper import WhisperModel

    # device="auto" uses CUDA if available, falls back to CPU
    # compute_type="int8" saves RAM on CPU (no quality loss for base/small)
    device = os.getenv("WHISPER_DEVICE", "auto")
    compute_type = "int8" if device in ("cpu", "auto") else "float16"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    if progress_callback:
        progress_callback(10, f"Transcribing with faster-whisper ({model_size})...")

    transcribe_kwargs = dict(
        word_timestamps=True,
        vad_filter=True,          # skip silence automatically
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    if language:
        transcribe_kwargs["language"] = language

    segments_gen, info = model.transcribe(file_path, **transcribe_kwargs)

    detected_lang = info.language

    if progress_callback:
        progress_callback(50, "Processing timestamps...")

    segments = []
    words = []
    full_text_parts = []

    for seg in segments_gen:
        full_text_parts.append(seg.text.strip())

        segments.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "speaker": None,  # filled by diarization below
            }
        )

        seg_words = seg.words or []
        if seg_words:
            for w in seg_words:
                words.append(
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "confidence": round(w.probability, 3),
                        "speaker": None,
                    }
                )
        else:
            # Fallback: distribute words evenly across segment duration
            text = seg.text.strip()
            if not text:
                continue
            word_list = text.split()
            seg_duration = seg.end - seg.start
            word_duration = seg_duration / max(len(word_list), 1)
            for i, word_text in enumerate(word_list):
                w_start = seg.start + i * word_duration
                words.append(
                    {
                        "word": word_text,
                        "start": round(w_start, 3),
                        "end": round(w_start + word_duration, 3),
                        "confidence": 0.5,
                        "speaker": None,
                    }
                )

    duration = info.duration or (segments[-1]["end"] if segments else 0)
    full_transcript = " ".join(full_text_parts)

    # ================================================================
    # Step 2: Speaker diarization (if enabled)
    # ================================================================
    speaker_segments = []
    speaker_summary = {"num_speakers": 0, "speakers": {}}
    diarization_warning = None

    if enable_diarization:
        try:
            from services.speaker_detection import (
                extract_audio_wav,
                run_diarization,
                assign_speakers_to_segments,
                assign_speakers_to_words,
                create_speaker_summary,
            )

            if progress_callback:
                progress_callback(55, "Extracting audio for speaker detection...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            try:
                extract_audio_wav(file_path, wav_path)

                if progress_callback:
                    progress_callback(60, "Running speaker diarization...")

                speaker_segments = run_diarization(
                    wav_path,
                    num_speakers=num_speakers,
                    progress_callback=lambda pct, msg: (
                        progress_callback(60 + int(pct * 0.3), msg) if progress_callback else None
                    ),
                )

                if speaker_segments:
                    if progress_callback:
                        progress_callback(92, "Assigning speakers to transcript...")

                    segments = assign_speakers_to_segments(segments, speaker_segments)
                    words = assign_speakers_to_words(words, speaker_segments)
                    speaker_summary = create_speaker_summary(speaker_segments)

                    if progress_callback:
                        progress_callback(
                            95,
                            f"Found {speaker_summary['num_speakers']} speakers",
                        )

            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

        except ImportError as e:
            diarization_warning = f"Speaker detection unavailable: {e}"
            if progress_callback:
                progress_callback(90, diarization_warning)
        except PermissionError as e:
            diarization_warning = str(e)
            if progress_callback:
                progress_callback(90, diarization_warning)
        except Exception as e:
            diarization_warning = f"Speaker detection failed: {e}"
            if progress_callback:
                progress_callback(90, diarization_warning)
    else:
        diarization_warning = "Speaker detection disabled"

    if progress_callback:
        progress_callback(100, "Transcription complete")

    result_data = {
        "transcript": full_transcript,
        "segments": segments,
        "words": words,
        "duration": round(duration, 3),
        "language": detected_lang,
        "speakers": speaker_summary,
        "speaker_segments": speaker_segments,
    }

    if diarization_warning:
        result_data["diarization_warning"] = diarization_warning

    return result_data
