#!/usr/bin/env python3
"""Simple voice recording script for TTS voice cloning."""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

SAMPLE_RATE = 24000  # Match Qwen3-TTS sample rate
DURATION = 10  # seconds


def record_voice(output_path: str = "my_voice.wav", duration: int = DURATION):
    """Record voice sample for TTS cloning."""
    
    print("\n" + "=" * 60)
    print("VOICE RECORDING FOR TTS CLONING")
    print("=" * 60)
    print(f"\nYou'll have {duration} seconds to speak.")
    print("\nSuggested script (or say anything natural):")
    print('-' * 60)
    print('"Hello, my name is [your name]. I\'m excited to share')
    print('information about Snowflake\'s community programs with you today."')
    print('-' * 60)
    print("\nTips:")
    print("  - Speak naturally at your normal pace")
    print("  - Stay close to the microphone")
    print("  - Avoid background noise")
    
    input("\nPress ENTER when ready to record...")
    
    print(f"\n>>> RECORDING for {duration} seconds... SPEAK NOW!")
    
    # Record audio
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to finish
    
    print(">>> Recording complete!")
    
    # Trim silence from beginning and end
    audio = audio.flatten()
    
    # Simple silence trimming (remove very quiet parts)
    threshold = 0.01
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) > 0:
        start = max(0, non_silent[0] - int(0.1 * SAMPLE_RATE))  # Keep 0.1s padding
        end = min(len(audio), non_silent[-1] + int(0.1 * SAMPLE_RATE))
        audio = audio[start:end]
    
    # Save to file
    output_path = Path(output_path)
    sf.write(output_path, audio, SAMPLE_RATE)
    
    duration_recorded = len(audio) / SAMPLE_RATE
    print(f"\nSaved: {output_path.absolute()}")
    print(f"Duration: {duration_recorded:.1f} seconds")
    
    # Register with TTS server
    register = input("\nRegister this voice with TTS server? [Y/n]: ").strip().lower()
    if register != 'n':
        import httpx
        voice_id = input("Voice ID (default: 'myvoice'): ").strip() or "myvoice"
        
        try:
            with open(output_path, 'rb') as f:
                response = httpx.post(
                    f"http://localhost:8001/register_voice_file?voice_id={voice_id}",
                    files={"file": (output_path.name, f, "audio/wav")},
                    timeout=30,
                )
            if response.status_code == 200:
                print(f"\nVoice registered as '{voice_id}'")
                print(f"To use it, set voice_id='{voice_id}' in TTS requests")
            else:
                print(f"\nFailed to register: {response.text}")
        except Exception as e:
            print(f"\nCouldn't connect to TTS server: {e}")
            print(f"You can manually register later with:")
            print(f'  curl -X POST "http://localhost:8001/register_voice_file?voice_id={voice_id}" -F "file=@{output_path}"')
    
    return str(output_path)


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "my_voice.wav"
    record_voice(output)
