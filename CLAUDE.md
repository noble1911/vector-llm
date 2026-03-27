# Claude Context File

> This file helps Claude understand the project and pick up where we left off.
> **Multiple Claude agents may work on this project in parallel.**

## ⚡ START HERE: Task Workflow

### When Starting a New Session

```bash
# 1. Check for available tasks
gh issue list

# 2. Claim an issue before starting work
gh issue edit <number> --add-assignee @me

# 3. Create a branch for your work
git checkout -b ron875/issue-<number>-short-description
```

### When Completing a Task

**Before creating a PR, always:**

1. **Review this file** — Does anything need updating based on what you learned?
2. **Create the PR:**
```bash
gh pr create --title "Short title" --body "Closes #N"
```

### When You Discover Undocumented Work

Create GitHub issues to track discoveries:
```bash
gh issue create --title "Short descriptive title" --body "## Task\n..."
```

---

## Project Overview

**Goal:** Replace Vector's clunky "Hey Vector" wake word interaction with an always-on LLM brain that enables natural conversation, autonomous vision-based movement, and proactive behaviour — all running locally on a Mac Mini M4.

**Owner:** Ron (GitHub: noble1911)
**Hardware:** Mac Mini M4 (24GB RAM, 512GB SSD) — also runs the home-server stack
**Repo:** github.com/noble1911/vector-llm
**Related:** github.com/noble1911/home-server (Butler API, Kokoro TTS, wire-pod)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Mac Mini M4                       │
│                                                      │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐  │
│  │ External  │───▶│ Whisper   │───▶│ Conversation │  │
│  │ Mic + VAD │    │ tiny STT  │    │ Manager      │  │
│  └──────────┘    └───────────┘    └──────┬───────┘  │
│                                          │           │
│  ┌──────────┐    ┌───────────┐    ┌──────▼───────┐  │
│  │ Vector   │───▶│ VLM       │───▶│ LLM Brain    │  │
│  │ Camera   │    │ (vision)  │    │ (Qwen2.5-3B) │  │
│  └──────────┘    └───────────┘    └──────┬───────┘  │
│                                          │           │
│  ┌──────────┐    ┌───────────┐    ┌──────▼───────┐  │
│  │ Vector   │◀───│ Kokoro    │◀───│ Vector SDK   │  │
│  │ Speaker  │    │ TTS       │    │ Control      │  │
│  └──────────┘    └───────────┘    └──────────────┘  │
│                                          │           │
│                                   ┌──────▼───────┐  │
│                                   │ Butler API   │  │
│                                   │ (escalation) │  │
│                                   └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Data Flow
1. **External mic** captures audio continuously (Vector's mic is poor quality)
2. **VAD** filters silence → **Whisper-tiny** transcribes speech segments
3. **Conversation manager** decides if speech is directed at Vector
4. **LLM brain** (Qwen2.5-3B via Ollama) reasons about what to say/do
5. **Vision loop** periodically feeds camera descriptions to the brain
6. **Vector SDK** executes movement, animations, expressions
7. **Kokoro TTS** generates speech → played on Vector's speaker
8. **Butler API** handles complex queries (escalation to Claude)

## Tech Stack

| Component | Technology | RAM Budget |
|-----------|-----------|------------|
| LLM Brain | Ollama + Qwen2.5-3B | ~2GB |
| Vision | Qwen2.5-VL-3B or moondream2 | ~1.5-2.5GB |
| STT | Whisper-tiny (faster-whisper) | ~150MB |
| TTS | Kokoro (already running in home-server) | 0 (shared) |
| VAD | silero-vad | ~50MB |
| Robot control | anki_vector SDK | minimal |
| Auth/tokens | wire-pod (native app on Mac) | 0 (native) |
| **Total new** | | **~4-5GB** |

## Project Structure

```
vector-llm/
├── CLAUDE.md                 # This file
├── README.md                 # User-facing overview
├── docker-compose.yml        # Ollama service
├── requirements.txt          # Python dependencies
├── config/
│   └── personality.yaml      # Vector personality prompt & settings
├── src/
│   ├── main.py               # Entry point — starts all loops
│   ├── brain.py              # LLM reasoning engine
│   ├── vision.py             # Camera capture + VLM processing
│   ├── stt.py                # Always-on speech-to-text
│   ├── tts.py                # Text-to-speech via Kokoro
│   ├── vector_control.py     # Vector SDK wrapper
│   ├── conversation.py       # Dialogue manager
│   ├── behaviors.py          # Autonomous idle behaviors
│   └── butler_client.py      # Butler API escalation
└── tests/
    └── ...
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Wake word | Eliminated | Core motivation — "Hey Vector" is annoying and unreliable |
| Audio input | External USB mic | Vector's mic is poor quality |
| LLM | Qwen2.5-3B via Ollama | Fits in RAM, fast on M4 Metal, good quality for size |
| Vision model | TBD (Qwen2.5-VL-3B or moondream2) | Need to test quality vs speed trade-off |
| STT | Whisper-tiny via faster-whisper | ~150MB, sub-second latency |
| TTS | Kokoro (shared with home-server) | Already deployed, good quality |
| Escalation | Butler API → Claude | Complex questions routed to full LLM |
| Robot auth | wire-pod tokens | Already running, no cloud dependency |

## Resource Context

The Mac Mini also runs the home-server stack (~3.5GB Docker containers). Current state:
- CPU: ~85% idle
- RAM: ~3.5GB used by Docker + system overhead
- Available for this project: ~8-10GB comfortably

## Conventions

- **Python 3.11+** with type hints
- **asyncio** for concurrent loops (STT, vision, brain, control)
- **Structured logging** with `structlog` or `logging`
- **Config via YAML** — personality, thresholds, model choices
- Commit style: imperative mood, brief description
- Git SSH alias: use `github.com` host (key: `~/.ssh/id_ed25519_github`)

## Notes for Future Sessions

- **Server is live** at 192.168.1.117 — wire-pod and all Docker stacks running
- **Kokoro TTS** is already running in the home-server voice-stack — reuse it, don't deploy a second instance
- **wire-pod** runs as a native macOS app at `/Applications/WirePod.app`
- **Vector SDK** needs tokens from wire-pod to authenticate
- **Always check `gh issue list` first** before starting work
- **Create issues for discovered work** — don't let insights get lost
- The local LLM handles casual conversation; complex queries escalate to Butler/Claude
- Vector's camera is 640x480 — optimize vision pipeline for low-res input
