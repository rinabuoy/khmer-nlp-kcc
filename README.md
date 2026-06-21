# khmer-nlp-kcc

Khmer NLP toolkit — word segmentation, POS tagging, sentiment polarity, and grapheme-to-phoneme conversion.

Powered by a custom Khmer language model. The NLP checkpoint is downloaded automatically from HuggingFace Hub on first use.

## Installation

```bash
pip install khmer-nlp-kcc
```

## Quick start

```python
from khmer_nlp import KhmerNLP

nlp = KhmerNLP()  # NLP checkpoint downloads on first call

# Word segmentation
nlp.segment("គាត់និយាយថា៖ការសិក្សានាំមកនូវចំណេះដឹងតើឯកភាពទេ?")
# → "គាត់ និយាយ ថា ៖ ការសិក្សា នាំ មក នូវ ចំណេះដឹង តើ ឯកភាព ទេ ?"

# Word segmentation with sub-word markers
nlp.segment("គាត់និយាយថា៖ការសិក្សានាំមកនូវចំណេះដឹងតើឯកភាពទេ?", markers=True)

# POS tagging
nlp.pos("គាត់និយាយថា៖ការសិក្សានាំមកនូវចំណេះដឹងតើឯកភាពទេ?")
# → [{"word": "គាត់", "label": "PRO"}, {"word": "និយាយ", "label": "VB"}, ...]

# Nova POS tagging
nlp.nova_pos("គាត់និយាយថា៖ការសិក្សានាំមកនូវចំណេះដឹងតើឯកភាពទេ?")
# → [{"word": "គាត់", "label": "o"}, {"word": "និយាយ", "label": "v"}, ...]

# Sentiment polarity
nlp.polarity("គាត់និយាយថា៖ការសិក្សានាំមកនូវចំណេះដឹងតើឯកភាពទេ?")
# → {"label": "neutral", "confidence": ..., "scores": {...}}

# All tasks at once
nlp.analyze("គាត់និយាយថា៖ការសិក្សានាំមកនូវចំណេះដឹងតើឯកភាពទេ?")
```

## Grapheme-to-phoneme (G2P)

The G2P model requires a separate checkpoint (`g2p_final_trans.pt`). Pass its path via `g2p_checkpoint_path`:

```python
nlp = KhmerNLP(g2p_checkpoint_path="/path/to/g2p_final_trans.pt")

# Phoneme sequence for a single word
nlp.g2p("ខ្មែរ")
# → ['kh', 'ae', '.', 'm', 'ae']

# Phoneme CER between two Khmer words
nlp.phoneme_cer("ខ្មែរ", "ខ្មែរភូមិ")
# → {
#     "word1": "ខ្មែរ",   "phones1": ["kh", "ae", ".", "m", "ae"],
#     "word2": "ខ្មែរភូមិ", "phones2": ["kh", "ae", ".", "m", "ae", ".", "ph", "uu", "m", ".", "m", "ɨ"],
#     "cer": 0.7
#   }
```

CER is computed at phoneme-token level (edit distance / number of phonemes in `word1`).

## Custom checkpoint

```python
nlp = KhmerNLP(checkpoint_path="/path/to/your/model.pt")
```

## Device selection

```python
import torch
nlp = KhmerNLP(device=torch.device("cuda:0"))
```

## Available methods

| Method | Returns | Description |
|--------|---------|-------------|
| `segment(text, *, markers=False)` | `str` | Space-joined segmented words; `markers=True` adds sub-word markers (`_`) |
| `pos(text)` | `list[dict]` | Word + POS label |
| `nova_pos(text)` | `list[dict]` | Word + Nova POS label |
| `polarity(text)` | `dict` | Sentiment polarity |
| `analyze(text)` | `dict` | All tasks combined |
| `g2p(word)` | `list[str]` | Phoneme token sequence for a Khmer word |
| `phoneme_cer(word1, word2)` | `dict` | Phoneme CER between two Khmer words |

Raw token-level outputs: `seg_tokens()`, `pos_tokens()`.

## License

MIT
