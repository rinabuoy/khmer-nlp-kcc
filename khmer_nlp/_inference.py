from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ._labels import SEG_LABELS, POS_LABELS, NOVA_LABELS

if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from ._models import KhmerMultiTaskClassifier


def _tokenize(text: str, tokenizer, device: torch.device, max_len: int = 512):
    enc = tokenizer.encode(text, add_special_tokens=False)
    BOS, EOS = tokenizer.bos_token_id, tokenizer.eos_token_id
    input_ids = ([BOS] + enc + [EOS])[:max_len]
    token_strs = tokenizer.convert_ids_to_tokens(enc)
    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    return ids_tensor, token_strs


@torch.no_grad()
def predict_token_labels(
    text: str,
    task: str,
    label_names: list,
    model: "KhmerMultiTaskClassifier",
    tokenizer: "AutoTokenizer",
    device: torch.device,
) -> list[dict]:
    ids, token_strs = _tokenize(text, tokenizer, device)
    logits = model(ids, task=task)
    preds = logits.argmax(-1)[0].tolist()[1: 1 + len(token_strs)]
    return [{"token": t, "label": label_names[p]} for t, p in zip(token_strs, preds)]


@torch.no_grad()
def classify_sentence(
    text: str,
    task: str,
    label_names: list,
    model: "KhmerMultiTaskClassifier",
    tokenizer: "AutoTokenizer",
    device: torch.device,
) -> dict:
    ids, _ = _tokenize(text, tokenizer, device)
    logits = model(ids, task=task)
    probs = F.softmax(logits[0], dim=-1).tolist()
    pred = int(logits[0].argmax(-1).item())
    return {
        "label":      label_names[pred],
        "confidence": round(probs[pred], 4),
        "scores":     {label_names[i]: round(p, 4) for i, p in enumerate(probs)},
    }


def _clean(tok: str) -> str:
    return tok.replace("▁", "").replace("Ġ", "") or tok


_SEG_SEP = {"I_PHRASE": "_", "I_COMP": "_", "I_DERIV": "_"}


def segment_words(text: str, model, tokenizer, device, *, markers: bool = False) -> list[str]:
    pairs = [(d["token"], d["label"]) for d in predict_token_labels(text, "seg", SEG_LABELS, model, tokenizer, device)]
    words: list[str] = []
    current: list[str] = []
    for token, label in pairs:
        if label == "SP":
            if current:
                words.append("".join(current))
                current = []
            continue
        if label == "B" or not current:
            if current:
                words.append("".join(current))
            current = [_clean(token)]
        elif markers and label in _SEG_SEP:
            current.append(_SEG_SEP[label] + _clean(token))
        else:
            current.append(_clean(token))
    if current:
        words.append("".join(current))
    return ' '.join([w for w in words if w])


def group_pos(text: str, model, tokenizer, device) -> list[dict]:
    seg_pairs = [(d["token"], d["label"]) for d in predict_token_labels(text, "seg", SEG_LABELS, model, tokenizer, device)]
    pos_pairs = [(d["token"], d["label"]) for d in predict_token_labels(text, "pos", POS_LABELS, model, tokenizer, device)]
    groups: list[dict] = []
    cur_chars: list[str] = []
    cur_pos = None
    for (tok, seg_label), (_, pos_label) in zip(seg_pairs, pos_pairs):
        if seg_label == "SP":
            if cur_chars:
                groups.append({"word": "".join(cur_chars), "label": cur_pos})
                cur_chars, cur_pos = [], None
            continue
        if seg_label == "B":
            if cur_chars:
                groups.append({"word": "".join(cur_chars), "label": cur_pos})
            cur_chars = [_clean(tok)]
            cur_pos = pos_label
        else:
            cur_chars.append(_clean(tok))
    if cur_chars:
        groups.append({"word": "".join(cur_chars), "label": cur_pos})
    return groups


def group_nova_pos(text: str, model, tokenizer, device) -> list[dict]:
    seg_pairs = [(d["token"], d["label"]) for d in predict_token_labels(text, "seg", SEG_LABELS, model, tokenizer, device)]
    nova_pairs = [(d["token"], d["label"]) for d in predict_token_labels(text, "nova", NOVA_LABELS, model, tokenizer, device)]
    groups: list[dict] = []
    cur_chars: list[str] = []
    cur_label = None
    for (tok, seg_label), (_, nova_label) in zip(seg_pairs, nova_pairs):
        if seg_label == "SP":
            if cur_chars:
                groups.append({"word": "".join(cur_chars), "label": cur_label})
                cur_chars, cur_label = [], None
            continue
        if seg_label == "B":
            if cur_chars:
                groups.append({"word": "".join(cur_chars), "label": cur_label})
            cur_chars = [_clean(tok)]
            cur_label = nova_label
        else:
            cur_chars.append(_clean(tok))
    if cur_chars:
        groups.append({"word": "".join(cur_chars), "label": cur_label})
    return groups
