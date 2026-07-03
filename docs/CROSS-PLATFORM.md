# Running Strata Voice on NVIDIA / AMD (Windows & Linux) — feasibility & plan

Status: **researched, not yet implemented.** This documents exactly what is
Apple-bound today, what the cross-platform replacements are, and the staged
plan to get there.

## The good news: most of the app is already portable

| Layer | Today | Portable? |
| :--- | :--- | :--- |
| **LLM (chat + memory)** | Ollama | ✅ Already ships native **NVIDIA (CUDA)** and **AMD (ROCm)** support on Windows/Linux — zero work |
| **Embeddings (recall)** | Ollama `nomic-embed-text` | ✅ Same — zero work |
| **Server / API** | Python stdlib | ✅ (one `strftime` portability bug already fixed) |
| **Frontend** | Single HTML file, Web Audio | ✅ Browser-portable |
| **Strata Memory** | SQLite | ✅ |
| **Audio I/O** | sounddevice / soundfile | ✅ Cross-platform (Linux may need ALSA/Pulse config) |
| **Keychain** | `keyring` | ✅ Has Windows Credential Manager / Secret Service backends |

**The entire port surface is three model loaders**, all currently on Apple's MLX:

1. **ASR** — Parakeet V3 via `mlx_audio.stt` (`load_asr()` in voicechat.py)
2. **TTS** — Kokoro-82M via `mlx_audio.tts` (+ `patch_kokoro_tts()`)
3. **VAD** — Silero via `mlx_audio.vad` (the micro-server in server.py)

## Each has a first-party cross-platform twin — same models, different runtime

- **Parakeet is natively an NVIDIA model.** `parakeet-tdt-0.6b-v3` is published
  by NVIDIA for [NeMo](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  (CUDA-first), and ONNX exports run fast on CPU (AVX2 + int8). Same weights,
  same quality we ship today.
- **Kokoro has an official ONNX distribution** —
  [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
  with the [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx) Python
  package: near-real-time on CPU, CUDA via `onnxruntime-gpu`, AMD via the
  DirectML EP (Windows) or ROCm EP (Linux).
- **Silero VAD's upstream distribution *is* ONNX/PyTorch** — mlx-audio's copy
  is the port. Trivial.

So the strategy is **one runtime (onnxruntime) for all three**, with execution
providers selecting the hardware: CUDA (NVIDIA), DirectML (AMD/any Windows GPU),
ROCm (AMD Linux), CPU (always works).

## Proposed architecture

A `speech.py` backend interface, selected once at startup:

```
class SpeechBackend:            # protocol
    def transcribe(path) -> Result      # .text
    def synthesize(text, voice, speed) -> np.ndarray
    def vad() -> StreamingVadLike       # .process(samples) -> events

MlxBackend    — today's code, unchanged (Apple Silicon)
OnnxBackend   — parakeet-onnx + kokoro-onnx + silero-onnx (everything else)
```

`server.py` and the VAD micro-server call the interface; platform detection
(`platform.system()`/`machine()`) picks the backend, with an env override
(`VOICE_SPEECH_BACKEND=mlx|onnx`). requirements split into
`requirements-mac.txt` / `requirements-cross.txt` (or extras:
`pip install .[mlx]` / `.[onnx]`), because mlx-audio must not be installed on
non-Mac and onnxruntime flavors differ per GPU vendor.

## Staged implementation (each stage independently shippable)

1. **VAD** (smallest): silero-onnx behind the interface; the TurnDetector
   state machine we wrote is pure Python and moves as-is. ~½ day.
2. **TTS**: kokoro-onnx behind the interface; voices/speed map 1:1; the
   Kokoro SineGen patch is MLX-only and simply doesn't apply. ~½–1 day.
3. **ASR**: parakeet ONNX (or NeMo on CUDA-rich machines); the `_ASR` wrapper
   already isolates the `.generate()/.text` shape. ~1 day incl. accuracy A/B.
4. **Installer**: `install.ps1` / Linux branch in install.sh, GPU detection
   (nvidia-smi / rocminfo), and picking the right onnxruntime package. ~1 day.

Total: roughly **3–4 focused days**, no architectural changes — the abstraction
seam already exists in miniature (`_ASR` wrapper, `_synth_sentence`, the VAD
handler's session object).

## Risks / open questions

- **Kokoro ONNX voice parity**: the `af_heart` etc. voice packs exist in the
  ONNX distribution, but cadence/quality should be A/B'd against MLX output.
- **espeak/misaki G2P on Windows**: kokoro-onnx bundles its own G2P path;
  verify the phonemizer story on a real Windows box before promising it.
- **Parakeet ONNX streaming**: batch transcription is proven; if we ever move
  to streaming ASR, NeMo (CUDA) is the safer path than ONNX.
- **Testing**: needs a real NVIDIA Windows box and ideally an AMD one; CPU-only
  fallback can be validated in CI/Linux VM.
