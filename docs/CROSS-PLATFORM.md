# Running Strata Voice on NVIDIA / AMD (Windows & Linux) — feasibility & plan

Status: **researched (fact-checked July 2026), not yet implemented.** This
documents exactly what is Apple-bound today, what the cross-platform
replacements are, and the staged plan to get there.

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
  (CUDA-first). For the portable path, don't hand-roll the export: the
  [`onnx-asr`](https://pypi.org/project/onnx-asr/) package wraps the maintained
  community export ([istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx))
  with one-line loading (`onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")`),
  multi-EP dispatch (CUDA / TensorRT / DirectML / MIGraphX / CoreML / CPU),
  quantized variants (int8, int4/int8 hybrid ~409 MB), and built-in long-form
  VAD — no PyTorch/Transformers dependency. Same weights, same quality we ship
  today. (A `parakeet-tdt-1.1b` exists for high-end cards; 0.6B stays the
  latency sweet spot. NeMo remains the native escape hatch on CUDA-rich
  machines, e.g. if we ever want true streaming ASR.)
- **Kokoro has an official ONNX distribution** —
  [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
  (still the canonical build; quantized fp16/q8/q4 variants available):
  near-real-time on CPU, GPU via the EPs below. Caveat: the
  [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx) wrapper package
  looks lightly maintained (v0.4.9, spring 2025) — evaluate at implementation
  time whether to use it, sherpa-onnx's packaging, or the onnx-community model
  directly with our own thin wrapper. (Kokoro v1.1 exists only as a
  Chinese-specific variant; v1.0 remains the English model.)
- **Silero VAD's upstream distribution *is* ONNX/PyTorch** — mlx-audio's copy
  is the port. Target the current **v6.2.1** ONNX weights (v6.x brought new
  ONNX models and made onnxruntime optional upstream), not whatever mlx-audio
  bundled. Trivial either way.

So the strategy is **one runtime (onnxruntime) for all three**, with execution
providers selecting the hardware:

| Hardware | Package | Execution provider |
| :--- | :--- | :--- |
| NVIDIA (Win/Linux) | `onnxruntime-gpu` | CUDA (TensorRT optional) |
| AMD Windows | `onnxruntime-directml` | DirectML ("sustained engineering" but fully functional; Windows ML → MIGraphX is the forward path on Win11 24H2+) |
| AMD Linux | MIGraphX-enabled ORT build (ROCm 7.0 stack) | **MIGraphX** — the dedicated ROCm EP was **removed in ONNX Runtime 1.23**; do not plan around it |
| Anything else | `onnxruntime` | CPU (always works) |

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
3. **ASR**: `onnx-asr` behind the interface (optional NeMo path on NVIDIA);
   the `_ASR` wrapper already isolates the `.generate()/.text` shape. ~1 day
   incl. accuracy A/B.
4. **Installer**: `install.ps1` / Linux branch in install.sh, GPU detection
   (`nvidia-smi` / `rocm-smi`/`amd-smi`), and picking the right onnxruntime
   flavor per the EP table above. ~1 day.

Total: roughly **3–4 focused days**, no architectural changes — the abstraction
seam already exists in miniature (`_ASR` wrapper, `_synth_sentence`, the VAD
handler's session object).

## Risks / open questions

- **Kokoro ONNX voice parity**: the `af_heart` etc. voice packs exist in the
  ONNX distribution, but cadence/quality should be A/B'd against MLX output.
- **kokoro-onnx wrapper maintenance**: lightly maintained; pick the packaging
  (wrapper vs sherpa-onnx vs direct onnxruntime) at implementation time.
- **espeak/misaki G2P on Windows**: the packaged G2P should handle it; verify
  once on a real Windows box before promising it.
- **Parakeet ONNX streaming**: batch transcription is proven (and sufficient
  for our VAD-segmented turns); for true streaming ASR later, NeMo (CUDA) is
  the safer path than ONNX.
- **AMD Linux stack**: MIGraphX EP requires the ROCm 7.0-era stack; expect the
  install story there to be the roughest of the four targets.
- **Testing**: needs a real NVIDIA Windows box and ideally AMD Windows + Linux;
  CPU-only fallback can be validated in CI/Linux VM.
