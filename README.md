# GhostMic — Real-time AI Meeting / Interview Assistant

> **Windows desktop application** — invisible to screen-capture software
> (Google Chrome `getDisplayMedia`, Zoom, Teams, OBS, Discord).

---

## Features

| Feature                     | Detail                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 🔇 Screen-share invisible   | `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` — window appears black/absent in all captures                          |
| 🎙 Dual audio capture       | WASAPI loopback (system audio) + microphone via `pyaudiowpatch` & `sounddevice`                                           |
| 🗣 Voice Activity Detection | Silero VAD — only transcribes actual speech, not silence                                                                  |
| 📝 Real-time transcription  | `faster-whisper` (base.en by default, CPU int8 or CUDA float16)                                                           |
| 🤖 AI suggestions           | Groq API (active provider)                                                                                                |
| 📄 Resume-aware assistance  | Upload PDF/DOCX/TXT resume, extract structured profile, and apply conservative resume-grounded transcript term correction |
| ⌨ Global hotkeys            | `Ctrl+Shift+G` record · `Ctrl+Shift+H` hide · `Ctrl+G` AI generate                                                        |
| 🔧 Settings dialog          | Tabs for Audio, AI, Appearance, Hotkeys — also stealthed                                                                  |
| 💾 Auto-save                | Transcript saved to `~/.ghostmic/` on exit                                                                                |

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
│   ├── ai_engine.py          # Groq AI response (QThread)
│   └── stealth.py            # SetWindowDisplayAffinity logic
├── services/
│   └── resume_service.py     # Resume ingestion, extraction, and structured profile persistence
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
│   ├── text_processing.py    # Whisper output cleanup
│   └── resume_context.py     # Resume-aware context summary and transcript correction helpers
└── assets/
    ├── icon.ico
    └── tray_icon.png
```

---

## Quick Start

### Requirements

- **Windows 10 version 2004 (build 19041)** or later for full
  capture-exclusion support. Older builds fall back to WDA_MONITOR
  (black rectangle in captures).
- Python 3.10+
- (Optional) CUDA GPU for faster transcription

### Installation

```bash
cd ghostmic
pip install -r requirements.txt
```

> **Note:** `pyaudiowpatch` is Windows-only. On other platforms the
> system-audio loopback capture will be skipped gracefully.

### Run

```bash
python ghostmic/main.py
# Or with debug logging:
python ghostmic/main.py --debug
# Start minimised to tray:
python ghostmic/main.py --minimized
```

### Build Windows EXE in CI

The repository includes a Windows build workflow at
`.github/workflows/build-windows.yml`.

- Every push/PR builds `dist/InterviewTool.exe` and uploads it as a workflow artifact.
- Pushing a tag like `v1.0.0` also creates a GitHub Release asset:
  `InterviewTool-v1.0.0-windows-x64.zip`.

Tag and push to publish a downloadable EXE:

```bash
git tag v1.0.0
git push origin v1.0.0
```

### First-time setup

1. Open **Settings** (gear icon or `⚙` in the overlay).
2. **Audio** tab → select your microphone and loopback device.
3. **AI** tab → paste your Groq API key from `console.groq.com`.
4. Click **OK** and press **⏺** (or `Ctrl+Shift+G`) to start recording.
5. Optional: **Resume** tab → upload your resume (PDF/DOCX/TXT) to enable resume-aware answer correction.

To re-enable OpenAI later, set `ai.expose_openai_provider` to `true` in `ghostmic/config.json`.

---

## Resume Upload + Resume-Aware Correction

GhostMic can ingest a resume and use structured resume facts during AI response generation.

### Supported resume formats

- `.pdf`
- `.docx`
- `.txt`

### Extraction schema (structured profile)

The app extracts and stores a profile including:

- identity details (name, email, phone, location)
- work history (company, role/title, dates, highlights)
- education
- skills
- projects
- certifications
- achievements
- companies, job titles, tools, technologies
- keywords and aliases for term matching

### End-to-end data flow

1. User uploads resume from **Settings → Resume**.
2. File is validated for extension and size.
3. Resume text is extracted and normalized.
4. Structured profile JSON is generated and persisted locally.
5. AI runtime receives only concise structured resume context (not full raw text).
6. For resume-related questions, AI treats structured resume facts as authoritative.
7. Transcript ambiguities are corrected conservatively when grounded by resume terms.

### Transcript correction policy

- High confidence: apply correction in interpretation context (e.g., `Micro hard` → `Microsoft` when resume-backed).
- Medium confidence: provide correction candidate with caveat.
- Low confidence: keep original transcript wording.
- Never invent details not present in the resume profile.
- Resume context should not dominate unrelated/general technical questions.

### Local storage and privacy

- Resume profile and extracted text are stored locally under `~/.ghostmic/resume/`.
- The AI prompt pipeline uses concise structured resume context, not the full raw resume body.
- Raw resume content should not be logged and should be treated as sensitive data.

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
