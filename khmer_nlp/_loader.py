from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from ._models import Decoder, KhmerLM, KhmerMultiTaskClassifier
from ._labels import (
    SEG_LABELS, POS_LABELS, NER_LABELS,
    NOVA_LABELS, PROFANITY_LABELS, POLARITY_LABELS,
)

HF_REPO_ID = "rinabuoy/khmer-nlp-kcc"
HF_FILENAME = "khmer_adapt_ner_nova_trans_ar_profanity_polarity.pt"
TOKENIZER_ID = "rinabuoy/khmer-latin-tokenizer-kcc"


def load(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> tuple[KhmerMultiTaskClassifier, AutoTokenizer]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    output_dim = len(tokenizer.get_vocab())
    pad_id = tokenizer.pad_token_id

    decoder = Decoder(output_dim, 512, 6, 8, 512 * 4, 0.1, device)
    lm = KhmerLM(decoder=decoder, trg_pad_idx=pad_id)
    model = KhmerMultiTaskClassifier(
        lm,
        num_pos_labels=len(POS_LABELS),
        num_seg_labels=len(SEG_LABELS),
        num_ner_labels=len(NER_LABELS),
        num_nova_word_labels=len(NOVA_LABELS),
        num_nova_inner_labels=len(NOVA_LABELS),
        num_profanity_labels=len(PROFANITY_LABELS),
        num_polarity_labels=len(POLARITY_LABELS),
    ).to(device)

    if checkpoint_path is None:
        local = Path(__file__).parent.parent / HF_FILENAME
        if local.exists():
            checkpoint_path = str(local)
        else:
            checkpoint_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                repo_type="model",
            )

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer
