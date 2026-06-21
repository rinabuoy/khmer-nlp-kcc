"""
khmer-nlp-kcc: Khmer word segmentation, POS tagging, NER, and sentiment polarity.
"""
from __future__ import annotations

from typing import Optional

import torch

from ._labels import SEG_LABELS, POS_LABELS, NOVA_LABELS, POLARITY_LABELS
from ._inference import (
    predict_token_labels,
    classify_sentence,
    segment_words,
    group_pos,
    group_nova_pos,
)
from ._g2p import load_g2p, g2p_word, phoneme_cer as _phoneme_cer

__version__ = "0.2.7"
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
        g2p_checkpoint_path: Optional[str] = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._g2p_checkpoint_path = g2p_checkpoint_path
        self._g2p_model = None

    def _ensure_loaded(self):
        if self._model is None:
            from ._loader import load
            self._model, self._tokenizer = load(self._checkpoint_path, self._device)

    def _ensure_g2p_loaded(self):
        if self._g2p_model is None:
            self._g2p_model = load_g2p(self._g2p_checkpoint_path, self._device)

    # ── Public API ─────────────────────────────────────────────────────────────

    def segment(self, text: str, *, markers: bool = False) -> str:
        """Return segmented Khmer words as a space-joined string."""
        self._ensure_loaded()
        return segment_words(text, self._model, self._tokenizer, self._device, markers=markers)

    def pos(self, text: str) -> list[dict]:
        """Return per-word POS tags: ``[{"word": ..., "label": ...}, ...]``."""
        self._ensure_loaded()
        return group_pos(text, self._model, self._tokenizer, self._device)

    def nova_pos(self, text: str) -> list[dict]:
        """Return per-word Nova POS tags: ``[{"word": ..., "label": ...}, ...]``."""
        self._ensure_loaded()
        return group_nova_pos(text, self._model, self._tokenizer, self._device)

    def g2p(self, word: str) -> list[str]:
        """Convert a Khmer word to a list of phoneme tokens."""
        self._ensure_g2p_loaded()
        return g2p_word(word, self._g2p_model, self._device)

    def phoneme_cer(self, word1: str, word2: str) -> dict:
        """Compute phoneme CER between two Khmer words.

        Returns the phoneme sequences and the CER score
        (edit distance at phoneme-token level / len(phones of word1)).
        """
        self._ensure_g2p_loaded()
        p1 = g2p_word(word1, self._g2p_model, self._device)
        p2 = g2p_word(word2, self._g2p_model, self._device)
        return {
            "word1":    word1,
            "phones1":  p1,
            "word2":    word2,
            "phones2":  p2,
            "cer":      _phoneme_cer(p1, p2),
        }

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
