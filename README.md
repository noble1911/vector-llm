# Vector LLM

Replace Vector's clunky "Hey Vector" wake word with an always-on LLM brain that enables natural conversation, autonomous vision-based movement, and proactive behaviour — all running locally on a Mac Mini M4.

## Goals

- **No wake word** — Vector listens continuously and responds naturally
- **Local inference** — all models run on-device (Qwen2.5-3B, Whisper-tiny, Kokoro TTS)
- **Vision-aware** — real-time object/motion detection with on-demand VLM
- **Autonomous behaviour** — idle animations, curiosity-driven movement
- **Escalation** — complex queries route to Claude via Butler API

## Hardware Requirements

- Mac Mini M4 (24GB RAM recommended)
- Anki Vector robot with wire-pod authentication
- External USB microphone (Vector's built-in mic is poor quality)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Mac Mini M4                          │
│                                                           │
│  ┌──────────┐    ┌───────────┐    ┌────────────────┐     │
│  │ External  │───▶│ Whisper   │───▶│ Conversation   │     │
│  │ Mic + VAD │    │ tiny STT  │    │ Manager        │     │
│  └──────────┘    └───────────┘    └───────┬────────┘     │
│                                           │               │
│  ┌──────────┐    ┌───────────┐    ┌───────▼────────┐     │
│  │ Vector   │───▶│ YOLO +    │───▶│ LLM Brain      │     │
│  │ Camera   │    │ OpenCV    │    │ (Qwen2.5-3B)   │     │
│  │ (5 fps)  │    │ (real-    │    │ text-only +    │     │
│  └──────────┘    │  time CV) │    │ tool use       │     │
│                  └─────┬─────┘    └──┬──────┬──────┘     │
│                        │             │      │             │
│                  ┌─────▼─────┐       │  ┌───▼──────────┐ │
│                  │ VLM       │◀──────┘  │ Vector SDK   │ │
│                  │ (on-demand│ "look"   │ Control      │ │
│                  │  tool)    │ tool call └──┬───────────┘ │
│                  └───────────┘             │              │
│  ┌──────────┐    ┌───────────┐    ┌───────▼────────┐     │
│  │ Vector   │◀───│ Kokoro    │◀───│ Butler API     │     │
│  │ Speaker  │    │ TTS       │    │ (escalation)   │     │
│  └──────────┘    └───────────┘    └────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

## Setup

### 1. Clone and create virtual environment

```bash
git clone git@github.com:noble1911/vector-llm.git
cd vector-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama

Ollama runs natively on macOS (Metal acceleration). For CI/portability, a Docker Compose file is provided:

```bash
# Native (recommended on Mac)
ollama serve
ollama pull qwen2.5:3b

# Or via Docker
docker compose up -d
```

### 3. Configure

Edit `config/personality.yaml` to adjust Vector's personality, model choices, and thresholds.

### 4. Run

```bash
python src/main.py
```

## Tech Stack

| Component | Technology | RAM Budget |
|-----------|-----------|------------|
| LLM Brain | Ollama + Qwen2.5-3B (text-only) | ~2GB |
| Real-time CV | YOLOv8-nano + OpenCV | ~4MB |
| VLM (on-demand) | Qwen2.5-VL-3B via Ollama | ~2GB (loaded on demand) |
| STT | Whisper-tiny (faster-whisper) | ~150MB |
| TTS | Kokoro (home-server) | 0 (shared) |
| VAD | silero-vad | ~50MB |
| Robot control | cyb3r_vector_sdk | minimal |

## Project Structure

```
vector-llm/
├── config/personality.yaml   # Personality prompt & settings
├── src/
│   ├── main.py               # Entry point
│   ├── brain.py              # LLM reasoning engine
│   ├── vision.py             # Real-time CV + on-demand VLM
│   ├── stt.py                # Speech-to-text
│   ├── tts.py                # Text-to-speech via Kokoro
│   ├── vector_control.py     # Vector SDK wrapper
│   ├── conversation.py       # Dialogue manager
│   ├── behaviors.py          # Autonomous idle behaviors
│   └── butler_client.py      # Butler API escalation
└── tests/
```

## License

Private project.
