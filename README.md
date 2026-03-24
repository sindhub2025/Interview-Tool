# GhostMic — Real-time AI Meeting / Interview Assistant

> **Windows desktop application** — invisible to screen-capture software
> (Google Chrome `getDisplayMedia`, Zoom, Teams, OBS, Discord).

---

## Features

| Feature | Detail |
|---|---|
| 🔇 Screen-share invisible | `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` — window appears black/absent in all captures |
| 🎙 Dual audio capture | WASAPI loopback (system audio) + microphone via `pyaudiowpatch` & `sounddevice` |
| 🗣 Voice Activity Detection | Silero VAD — only transcribes actual speech, not silence |
| 📝 Real-time transcription | `faster-whisper` (base.en by default, CPU int8 or CUDA float16) |
| 🤖 AI suggestions | Groq API (free tier) **or** Ollama (fully local) |
| ⌨ Global hotkeys | `Ctrl+Shift+G` record · `Ctrl+Shift+H` hide · `Ctrl+G` AI generate |
| 🔧 Settings dialog | Tabs for Audio, AI, Appearance, Hotkeys — also stealthed |
| 💾 Auto-save | Transcript saved to `~/.ghostmic/` on exit |

---

## Project Structure

```
ghostmic/
├── main.py                   # Entry point
├── requirements.txt
├── config.json               # User settings
├── core/
│   ├── audio_buffer.py       # Thread-safe circular audio buffer
│   ├── audio_capture.py      # WASAPI loopback + mic capture (QThread)
│   ├── vad.py                # Silero VAD state machine (QThread)
│   ├── transcription_engine.py  # faster-whisper (QThread)
│   ├── ai_engine.py          # Groq / Ollama AI response (QThread)
│   └── stealth.py            # SetWindowDisplayAffinity logic
├── ui/
│   ├── main_window.py        # Frameless overlay window
│   ├── transcript_panel.py   # Auto-scrolling transcript bubbles
│   ├── ai_response_panel.py  # Streaming AI response cards
│   ├── controls_bar.py       # Record / mode / settings controls
│   ├── settings_dialog.py    # Multi-tab settings (also stealthed)
│   ├── system_tray.py        # System tray icon & menu
│   └── styles.py             # Dark QSS theme
├── utils/
│   ├── hotkeys.py            # Global hotkey registration (pynput)
│   ├── logger.py             # Rotating file logger
│   └── text_processing.py    # Whisper output cleanup
└── assets/
    ├── icon.ico
    └── tray_icon.png
```

---

## Quick Start

### Requirements
- **Windows 10 version 2004 (build 19041)** or later for full
  capture-exclusion support.  Older builds fall back to WDA_MONITOR
  (black rectangle in captures).
- Python 3.10+
- (Optional) CUDA GPU for faster transcription

### Installation

```bash
cd ghostmic
pip install -r requirements.txt
```

> **Note:** `pyaudiowpatch` is Windows-only.  On other platforms the
> system-audio loopback capture will be skipped gracefully.

### Run

```bash
python ghostmic/main.py
# Or with debug logging:
python ghostmic/main.py --debug
# Start minimised to tray:
python ghostmic/main.py --minimized
```

### First-time setup

1. Open **Settings** (gear icon or `⚙` in the overlay).
2. **Audio** tab → select your microphone and loopback device.
3. **AI** tab → paste your free [Groq API key](https://console.groq.com)
   **or** choose *Ollama* if you have it installed locally.
4. Click **OK** and press **⏺** (or `Ctrl+Shift+G`) to start recording.

---

## Stealth Implementation

```python
# core/stealth.py
WDA_EXCLUDEFROMCAPTURE = 0x00000011

def apply_stealth(hwnd: int) -> bool:
    ok = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    if not ok:
        # Fallback: WDA_MONITOR (shows black, not captured content)
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000001)
    return ok
```

- Applied **after** the window has a valid HWND (`QTimer.singleShot(100, …)`)
- Re-applied on `changeEvent` (minimize/restore) and `showEvent`
- Applied to **all** windows: main overlay, settings dialog
- Verified at startup using `mss` screenshot comparison

---

## Threading Architecture

```
Main Thread (PyQt6 event loop)
  │
  ├── SystemAudioCaptureThread  ──push_chunk()──►
  ├── MicCaptureThread          ──push_chunk()──►  VADThread
  │                                                     │
  │                                              speech_segment_ready
  │                                                     │
  │                                          TranscriptionThread
  │                                                     │
  │                                          transcription_ready → UI
  │                                                     │
  │                                               AIThread
  │                                                     │
  │                                          ai_response_chunk/ready → UI
```

All cross-thread communication uses Qt Signals/Slots (thread-safe).

---

## License

MIT