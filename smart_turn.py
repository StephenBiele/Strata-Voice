"""Semantic endpointing — Smart Turn v3.2 (ONNX).

Decides whether a user's utterance is a *complete* thought (respond now) or a
trailing-off pause mid-sentence (keep listening), from prosody + content —
instead of ending the turn on a fixed silence timer. This is what lets the
assistant be snappy without cutting you off: it waits when you're clearly
mid-thought and replies the instant you're actually done.

Runs on CPU via onnxruntime (~8 ms/check), so it lives on the VAD micro-server
thread and never touches the MLX Metal stream. Model is ~8.7 MB, lazy-downloaded
and cached by huggingface_hub on first use.

Model card: https://huggingface.co/pipecat-ai/smart-turn-v3 (BSD-2-Clause).
Returns P(complete) in [0, 1]; the caller compares against a tunable threshold.
"""
from __future__ import annotations

import numpy as np

_REPO = "pipecat-ai/smart-turn-v3"
_FILE = "smart-turn-v3.2-cpu.onnx"
_SR = 16000
_WIN = 8 * _SR  # the model's context is the last 8 seconds of speech

_sess = None
_fe = None


def load() -> None:
    """Download (first run) and load the ONNX session + feature extractor."""
    global _sess, _fe
    if _sess is not None:
        return
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(_REPO, _FILE)
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1  # single-turn checks; keep it light
    _sess = ort.InferenceSession(path, sess_options=so)
    _fe = WhisperFeatureExtractor(chunk_length=8)


def predict(samples16k: np.ndarray) -> float:
    """P(turn complete) for float32 mono audio @ 16 kHz in [-1, 1].

    Uses the last 8 s (padded if shorter) — the same window the model trains on.
    """
    if _sess is None:
        load()
    a = samples16k[-_WIN:] if len(samples16k) > _WIN else samples16k
    inp = _fe(a, sampling_rate=_SR, return_tensors="np", padding="max_length",
              max_length=_WIN, truncation=True, do_normalize=True)
    feats = np.expand_dims(inp.input_features.squeeze(0).astype(np.float32), 0)
    out = _sess.run(None, {_sess.get_inputs()[0].name: feats})
    return float(np.array(out[0]).reshape(-1)[0])
