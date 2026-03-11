"""TTS Service API for Snowpark Container Services.

This service provides voice cloning TTS via Qwen3-TTS, accessible via HTTP API.
Deploy to SPCS with GPU compute for best performance.
Uses MLX on Apple Silicon for fast local inference.
"""

import io
import os
import base64
import logging
import platform
import tempfile
import json
from pathlib import Path
from typing import Optional, AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen3-TTS Voice Cloning Service",
    description="Text-to-speech with voice cloning, powered by Qwen3-TTS",
    version="1.0.0",
)

# Global model instance (loaded once on startup)
_tts_model = None
_use_mlx = False  # Whether we're using MLX (Apple Silicon) or PyTorch
_voice_data: dict[str, dict] = {}  # Cache voice path and ref_text by name

# Default ref_text for voice cloning - should match what's in the voice sample
DEFAULT_REF_TEXT = "Hello, my name is James and I work at Snowflake. I'm excited to help you learn about our community programs today. Whether you're interested in becoming a Data Superhero, joining our Champions program, or attending one of our many events, I'm here to guide you through it all."

# Pitch shift in semitones (positive = higher, negative = lower)
# Adjust this to match your voice pitch
PITCH_SHIFT_SEMITONES = 3  # Raise pitch by 3 semitones


class TTSRequest(BaseModel):
    """Request body for TTS generation."""

    text: str
    voice_id: str = "default"  # ID of registered voice
    temperature: float = 0.3
    top_p: float = 0.95


class TTSResponse(BaseModel):
    """Response with generated audio."""

    audio_base64: str
    sample_rate: int
    duration_seconds: float


class VoiceRegisterRequest(BaseModel):
    """Request to register a new voice from base64 audio."""

    voice_id: str
    audio_base64: str  # WAV file as base64


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    registered_voices: list[str]


def get_model():
    """Get or initialize the TTS model."""
    global _tts_model, _use_mlx

    if _tts_model is None:
        logger.info("Loading Qwen3-TTS model...")

        # Check if we're on Apple Silicon - use MLX for much faster inference
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        
        if is_apple_silicon:
            try:
                from mlx_audio.tts.utils import load_model
                logger.info("Using MLX backend (Apple Silicon)")
                # Use Base model for voice cloning support (CustomVoice doesn't support ref_audio)
                # 1.7B model has better quality than 0.6B
                _tts_model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit")
                _use_mlx = True
                logger.info("Qwen3-TTS 1.7B Base MLX 4-bit model loaded successfully (supports voice cloning)")
                return _tts_model
            except ImportError:
                logger.warning("mlx-audio not installed, falling back to PyTorch")
        
        # Fallback to PyTorch for CUDA or CPU
        import torch
        from qwen_tts import Qwen3TTSModel

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
        logger.info(f"Using PyTorch device: {device}")

        _tts_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map=device,
            dtype=dtype,
        )
        _use_mlx = False
        logger.info("Qwen3-TTS 0.6B CustomVoice model loaded successfully")

    return _tts_model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        get_model()
        logger.info("TTS service ready")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't fail startup - model will be loaded on first request


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_tts_model is not None,
        registered_voices=list(_voice_data.keys()),
    )


@app.post("/register_voice")
async def register_voice(request: VoiceRegisterRequest):
    """Register a voice sample for cloning.

    The voice sample should be 3-10 seconds of clear speech.
    """
    try:
        import soundfile as sf
        import tempfile

        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Store the voice sample path for this voice_id
            # In production, you'd store this in a stage or extract embeddings
            voice_dir = Path("/tmp/voices")
            voice_dir.mkdir(exist_ok=True)
            voice_path = voice_dir / f"{request.voice_id}.wav"

            # Copy to persistent location
            import shutil
            shutil.copy(tmp_path, voice_path)

            _voice_data[request.voice_id] = {"path": str(voice_path), "ref_text": DEFAULT_REF_TEXT}
            logger.info(f"Registered voice: {request.voice_id}")

            return {"status": "success", "voice_id": request.voice_id}

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Failed to register voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register_voice_file")
async def register_voice_file(voice_id: str, file: UploadFile = File(...)):
    """Register a voice sample via file upload."""
    try:
        voice_dir = Path("/tmp/voices")
        voice_dir.mkdir(exist_ok=True)
        voice_path = voice_dir / f"{voice_id}.wav"

        # Save uploaded file
        content = await file.read()
        with open(voice_path, "wb") as f:
            f.write(content)

        _voice_data[voice_id] = {"path": str(voice_path), "ref_text": DEFAULT_REF_TEXT}
        logger.info(f"Registered voice from file: {voice_id}")

        return {"status": "success", "voice_id": voice_id}

    except Exception as e:
        logger.error(f"Failed to register voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """Generate speech from text using a registered voice."""
    try:
        model = get_model()

        # Get voice data (path and ref_text)
        voice_info = _voice_data.get(request.voice_id)
        voice_path = voice_info["path"] if voice_info else None
        ref_text = voice_info.get("ref_text", DEFAULT_REF_TEXT) if voice_info else None
        logger.info(f"Request voice_id: {request.voice_id}, voice_path: {voice_path}")
        logger.info(f"Registered voices: {list(_voice_data.keys())}")
        if not voice_path and request.voice_id != "default":
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{request.voice_id}' not registered. Use /register_voice first.",
            )

        logger.info(f"Generating speech for: {request.text[:50]}...")

        if _use_mlx:
            # MLX backend (Apple Silicon) - much faster
            from mlx_audio.tts.generate import generate_audio
            import soundfile as sf
            
            # Generate to temp file
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_prefix = os.path.join(tmp_dir, "output")
                
                # For voice cloning with Base model, we need ref_audio + ref_text
                # The ref_text should match what was said in the voice sample
                if voice_path:
                    logger.info(f"Using voice cloning with ref_audio: {voice_path}")
                    generate_audio(
                        model=model,
                        text=request.text,
                        ref_audio=voice_path,
                        ref_text=ref_text,
                        file_prefix=file_prefix,
                        verbose=True,
                    )
                else:
                    # No voice cloning - use default Base model voice
                    generate_audio(
                        model=model,
                        text=request.text,
                        voice="af_heart",  # Base model default voice
                        file_prefix=file_prefix,
                        verbose=True,
                    )
                    
                # Read the generated WAV file
                wav_path = f"{file_prefix}_000.wav"
                audio, sample_rate = sf.read(wav_path)
        else:
            # PyTorch backend (CUDA/CPU)
            if voice_path:
                # Voice cloning with registered voice sample
                wavs, sample_rate = model.generate_voice_clone(
                    text=request.text,
                    speaker_wav=voice_path,
                    top_p=request.top_p,
                )
            else:
                # Use built-in custom voice speaker
                wavs, sample_rate = model.generate_custom_voice(
                    text=request.text,
                    language="Auto",
                    speaker="Vivian",  # Default English female voice
                )
            # Get first audio from batch
            audio = wavs[0] if isinstance(wavs, list) else wavs

        # Ensure numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        # Apply pitch shift if configured
        if PITCH_SHIFT_SEMITONES != 0:
            import librosa
            audio = librosa.effects.pitch_shift(
                audio, 
                sr=sample_rate, 
                n_steps=PITCH_SHIFT_SEMITONES
            )
            logger.info(f"Applied pitch shift: {PITCH_SHIFT_SEMITONES:+d} semitones")

        # Normalize audio volume to consistent level
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 0.9  # Target peak amplitude
            audio = audio * (target_peak / peak)

        # Convert to int16 for transmission
        audio_int16 = (audio * 32767).astype(np.int16)

        # Encode as base64
        audio_base64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

        duration = len(audio) / sample_rate
        logger.info(f"Generated {duration:.2f}s of audio")

        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=sample_rate,
            duration_seconds=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_wav")
async def synthesize_wav(request: TTSRequest):
    """Generate speech and return as WAV file."""
    try:
        import soundfile as sf

        # Get the audio data
        response = await synthesize(request)

        # Decode audio
        audio_bytes = base64.b64decode(response.audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767

        # Create WAV in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio, response.sample_rate, format="WAV")
        buffer.seek(0)

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"WAV synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_stream")
async def synthesize_stream(request: TTSRequest):
    """Generate speech with streaming - returns audio chunks as they're generated.
    
    Uses Server-Sent Events (SSE) format. Each event contains:
    - event: "audio" for audio chunks, "done" when complete
    - data: JSON with audio_base64, sample_rate, chunk_index, is_final
    
    This reduces time-to-first-audio from ~3-4s to ~0.7s.
    """
    
    async def generate_chunks() -> AsyncGenerator[str, None]:
        try:
            model = get_model()
            
            # Get voice data
            voice_info = _voice_data.get(request.voice_id)
            voice_path = voice_info["path"] if voice_info else None
            ref_text = voice_info.get("ref_text", DEFAULT_REF_TEXT) if voice_info else None
            
            if not voice_path and request.voice_id != "default":
                error_data = json.dumps({"error": f"Voice '{request.voice_id}' not registered"})
                yield f"event: error\ndata: {error_data}\n\n"
                return
            
            logger.info(f"Streaming speech for: {request.text[:50]}...")
            
            if not _use_mlx:
                # Non-MLX doesn't support streaming, fall back to full generation
                error_data = json.dumps({"error": "Streaming only supported on MLX backend"})
                yield f"event: error\ndata: {error_data}\n\n"
                return
            
            import librosa
            
            chunk_index = 0
            first_chunk_time = None
            
            # Use model.generate with streaming
            for result in model.generate(
                text=request.text,
                ref_audio=voice_path,
                ref_text=ref_text,
                stream=True,
                streaming_interval=0.5,  # Smaller chunks for faster first response
                verbose=False,
            ):
                if hasattr(result, 'audio') and result.audio is not None:
                    audio = np.array(result.audio, dtype=np.float32)
                    sample_rate = result.sample_rate
                    
                    # Apply pitch shift
                    if PITCH_SHIFT_SEMITONES != 0:
                        audio = librosa.effects.pitch_shift(
                            audio,
                            sr=sample_rate,
                            n_steps=PITCH_SHIFT_SEMITONES
                        )
                    
                    # Apply noise gate to remove low-level static
                    noise_threshold = 0.02  # Threshold below which audio is considered noise
                    audio = np.where(np.abs(audio) < noise_threshold, 0, audio)
                    
                    # Normalize chunk
                    peak = np.max(np.abs(audio))
                    if peak > 0:
                        audio = audio * (0.9 / peak)
                    
                    # Apply short fade-in/fade-out to reduce artifacts at chunk boundaries
                    fade_samples = min(int(sample_rate * 0.01), len(audio) // 4)  # 10ms fade or 1/4 of chunk
                    if fade_samples > 0 and len(audio) > fade_samples * 2:
                        # Fade in at start
                        fade_in = np.linspace(0, 1, fade_samples)
                        audio[:fade_samples] *= fade_in
                        # Fade out at end
                        fade_out = np.linspace(1, 0, fade_samples)
                        audio[-fade_samples:] *= fade_out
                    
                    # Convert to int16
                    audio_int16 = (audio * 32767).astype(np.int16)
                    audio_base64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
                    
                    chunk_data = json.dumps({
                        "audio_base64": audio_base64,
                        "sample_rate": sample_rate,
                        "chunk_index": chunk_index,
                        "duration_seconds": len(audio) / sample_rate,
                        "is_final": False,
                    })
                    
                    yield f"event: audio\ndata: {chunk_data}\n\n"
                    chunk_index += 1
            
            # Send completion event
            done_data = json.dumps({"chunk_index": chunk_index, "is_final": True})
            yield f"event: done\ndata: {done_data}\n\n"
            
            logger.info(f"Streamed {chunk_index} chunks")
            
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"
    
    return StreamingResponse(
        generate_chunks(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/sf_synthesize")
async def sf_synthesize(request: dict):
    """Snowflake service function compatible TTS endpoint.
    
    Expects: {"data": [[row_index, text, voice_id], ...]}
    Returns: {"data": [[row_index, {"audio_base64": ..., "sample_rate": ..., "duration_seconds": ...}], ...]}
    """
    try:
        results = []
        for row in request.get("data", []):
            row_index = row[0]
            text = row[1] if len(row) > 1 else ""
            voice_id = row[2] if len(row) > 2 else "default"
            
            # Create a TTSRequest and call synthesize
            tts_request = TTSRequest(text=text, voice_id=voice_id)
            response = await synthesize(tts_request)
            
            results.append([row_index, {
                "audio_base64": response.audio_base64,
                "sample_rate": response.sample_rate,
                "duration_seconds": response.duration_seconds,
            }])
        
        return {"data": results}
    
    except Exception as e:
        logger.error(f"Snowflake synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
