# Troubleshooting Guide

## Robot is Completely Silent (No Audio Output)

### Symptoms
- App starts successfully, OpenAI connects
- Robot may do dance moves or other actions
- But robot never speaks - completely silent

### Root Cause
The microphone is not capturing audio (returns all zeros). This prevents OpenAI from detecting speech and responding with audio.

### Diagnosis

**Step 1: Test if microphone is working**
```bash
cd /Users/jchaearley/Snowflake/community_bar/reachy_mini_conversation_app
source .venv/bin/activate
python3 -c "
import sounddevice as sd
import numpy as np
recording = sd.rec(int(1 * 16000), samplerate=16000, channels=1, dtype='float32', device=3)
sd.wait()
max_val = np.abs(recording).max()
print(f'Max audio level: {max_val}')
if max_val > 0.001:
    print('Microphone is working!')
else:
    print('Microphone returning zeros - see fix below')
"
```

If max audio level is 0.0, the microphone is blocked.

### Fix

**Step 1: Kill Reachy Mini Control App**

Check if the Reachy Mini Control app is running:
```bash
ps aux | grep -i reachy | grep -v grep
```

If running, kill it:
```bash
pkill -f "reachy-mini-control"
```

**Step 2: Restart CoreAudio Daemon**
```bash
sudo killall coreaudiod
```

This restarts macOS audio system and releases any stuck audio devices.

**Step 3: Verify Fix**

Run the microphone test again (Step 1 in Diagnosis). You should see a non-zero max audio level.

**Step 4: Start Conversation App**
```bash
cd /Users/jchaearley/Snowflake/community_bar/reachy_mini_conversation_app
source .venv/bin/activate
python -c "from reachy_mini_conversation_app.main import main; main()"
```

### Alternative Fixes

If the above doesn't work:
1. **Unplug and replug** the Reachy Mini USB cable
2. **Reboot** the computer (nuclear option, but always works)

### Quick One-Liner Fix

```bash
pkill -f "reachy-mini-control"; sleep 2; sudo killall coreaudiod
```

---

## Other Issues

### Profile Not Loading

If the robot isn't using the Snowflake Community profile, check:
```bash
cat .env | grep REACHY_MINI_CUSTOM_PROFILE
```

Should show:
```
REACHY_MINI_CUSTOM_PROFILE="snowflake_community"
```

### Robot Daemon Not Running

If you see connection errors like `Unable to connect to any of [tcp/localhost:7447]`:
1. Check if daemon is running: `systemctl status reachy-mini-daemon`
2. Restart daemon: `systemctl restart reachy-mini-daemon`
3. Or reboot the Reachy Mini robot
