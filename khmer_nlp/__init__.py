"""
khmer-nlp-kcc: Khmer word segmentation, POS tagging, NER, and sentiment polarity.
"""
from __future__ import annotations

from typing import Optional

import torch

from ._labels import SEG_LABELS, POS_LABELS, POLARITY_LABELS
from ._inference import (
    predict_token_labels,
    classify_sentence,
    segment_words,
    group_pos,
)

__version__ = "0.2.3"
__all__ = ["KhmerNLP"]


class KhmerNLP:
    """
    Khmer NLP toolkit — word segmentation, POS tagging, and sentiment polarity.

    The model is lazy-loaded on the first inference call.

    Parameters
    ----------
    checkpoint_path:
        Path to a local ``.pt`` checkpoint.  When omitted the checkpoint is
        downloaded automatically from HuggingFace Hub
        (``rinabuoy/khmer-ocr-checkpoints``).
    device:
        ``torch.device`` to run inference on.  Defaults to CUDA when available.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            from ._loader import load
            self._model, self._tokenizer = load(self._checkpoint_path, self._device)

    # ── Public API ─────────────────────────────────────────────────────────────

    def segment(self, text: str) -> list[str]:
        """Return a list of segmented Khmer words."""
        self._ensure_loaded()
        return segment_words(text, self._model, self._tokenizer, self._device)

    def pos(self, text: str) -> list[dict]:
        """Return per-word POS tags: ``[{"word": ..., "label": ...}, ...]``."""
        self._ensure_loaded()
        return group_pos(text, self._model, self._tokenizer, self._device)

    def polarity(self, text: str) -> dict:
        """Sentence-level sentiment polarity classification."""
        self._ensure_loaded()
        return classify_sentence(text, "polarity", POLARITY_LABELS, self._model, self._tokenizer, self._device)

    def seg_tokens(self, text: str) -> list[dict]:
        """Raw token-level segmentation labels."""
        self._ensure_loaded()
        return predict_token_labels(text, "seg", SEG_LABELS, self._model, self._tokenizer, self._device)

    def pos_tokens(self, text: str) -> list[dict]:
        """Raw token-level POS labels."""
        self._ensure_loaded()
        return predict_token_labels(text, "pos", POS_LABELS, self._model, self._tokenizer, self._device)

    def analyze(self, text: str) -> dict:
        """Run seg, pos, and polarity together and return a combined result dict."""
        self._ensure_loaded()
        text = text.strip()
        return {
            "text":     text,
            "words":    self.segment(text),
            "pos":      self.pos(text),
            "polarity": self.polarity(text),
        }
