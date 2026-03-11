#!/usr/bin/env python3
"""Record an optimized voice sample for Qwen3-TTS voice cloning."""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os

# The exact text you should say - this will be used as ref_text
SCRIPT = """Hello, my name is James and I work at Snowflake. 
I'm excited to help you learn about our community programs today. 
Whether you're interested in becoming a Data Superhero, joining our Champions program, 
or attending one of our many events, I'm here to guide you through it all."""

# Clean version for ref_text (single line, normalized spacing)
REF_TEXT = " ".join(SCRIPT.split())

SAMPLE_RATE = 24000  # Qwen3-TTS native sample rate
DURATION = 20  # seconds - enough time to say the script naturally


def choose_input_device():
    """Let user choose which microphone to use."""
    print("Available input devices:")
    devices = sd.query_devices()
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = "* " if i == sd.default.device[0] else "  "
            print(f"  {marker}[{i}] {d['name']} ({d['max_input_channels']} channels)")
            input_devices.append(i)
    
    print()
    choice = input(f"Enter device number (or press ENTER for default [{sd.default.device[0]}]): ").strip()
    
    if choice == "":
        return sd.default.device[0]
    
    try:
        device_id = int(choice)
        if device_id in input_devices:
            return device_id
        else:
            print(f"Invalid device. Using default.")
            return sd.default.device[0]
    except ValueError:
        print(f"Invalid input. Using default.")
        return sd.default.device[0]

def record_voice():
    print("=" * 60)
    print("VOICE SAMPLE RECORDING FOR QWEN3-TTS")
    print("=" * 60)
    print()
    
    # Choose microphone
    device_id = choose_input_device()
    device_name = sd.query_devices(device_id)['name']
    print(f"\nUsing: {device_name}")
    print()
    
    print("Please read the following script naturally and clearly:")
    print("-" * 60)
    print(SCRIPT)
    print("-" * 60)
    print()
    print("Tips for best results:")
    print("  - Speak at your normal pace")
    print("  - Use natural intonation (not monotone)")
    print("  - Keep a consistent distance from the mic")
    print("  - Minimize background noise")
    print()
    
    input("Press ENTER when ready to record...")
    print()
    print(f"Recording for {DURATION} seconds...")
    print("START SPEAKING NOW!")
    print()
    
    # Record with selected device
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=device_id)
    sd.wait()
    audio = audio.flatten()
    
    print("Recording complete!")
    print()
    
    # Normalize audio
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.9 / peak)
    
    # Trim silence from start and end
    threshold = 0.02
    non_silent = np.abs(audio) > threshold
    if np.any(non_silent):
        start = np.argmax(non_silent)
        end = len(audio) - np.argmax(non_silent[::-1])
        # Add small padding
        padding = int(0.1 * SAMPLE_RATE)
        start = max(0, start - padding)
        end = min(len(audio), end + padding)
        audio = audio[start:end]
    
    # Stats
    duration = len(audio) / SAMPLE_RATE
    rms = np.sqrt(np.mean(audio**2))
    print(f"Processed audio: {duration:.1f}s, RMS: {rms:.3f}")
    
    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "voice_sample_optimized.wav")
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Saved to: {output_path}")
    
    # Also save to profile
    profile_path = os.path.join(
        os.path.dirname(output_dir),
        "src/reachy_mini_conversation_app/profiles/snowflake_community/voice_sample.wav"
    )
    sf.write(profile_path, audio, SAMPLE_RATE)
    print(f"Saved to profile: {profile_path}")
    
    # Print the ref_text to use
    print()
    print("=" * 60)
    print("REF_TEXT (copy this to app.py):")
    print("=" * 60)
    print(f'ref_text = "{REF_TEXT}"')
    print()
    
    # Register with server
    register = input("Register with TTS server now? [Y/n]: ").strip().lower()
    if register != 'n':
        import requests
        try:
            with open(output_path, 'rb') as f:
                resp = requests.post(
                    "http://localhost:8001/register_voice_file?voice_id=cartoon",
                    files={"file": f}
                )
            if resp.status_code == 200:
                print("✓ Voice registered successfully!")
            else:
                print(f"✗ Registration failed: {resp.text}")
        except Exception as e:
            print(f"✗ Could not connect to server: {e}")
    
    return output_path, REF_TEXT

if __name__ == "__main__":
    record_voice()
