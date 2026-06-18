# khmer-nlp-kcc

Khmer NLP toolkit — word segmentation, POS tagging, and sentiment polarity.

Powered by a custom Khmer language model. The checkpoint is downloaded automatically from HuggingFace Hub on first use.

## Installation

```bash
pip install khmer-nlp-kcc
```

## Quick start

```python
from khmer_nlp import KhmerNLP

nlp = KhmerNLP()  # checkpoint downloads on first call

# Word segmentation
nlp.segment("គាត់ចូលចិត្តអានសៀវភៅ")
# → ["គាត់", "ចូលចិត្ត", "អាន", "សៀវភៅ"]

# POS tagging
nlp.pos("គាត់ចូលចិត្តអានសៀវភៅ")
# → [{"word": "គាត់", "label": "PRO"}, {"word": "ចូលចិត្ត", "label": "VB"}, {"word": "អាន", "label": "VB"}, {"word": "សៀវភៅ", "label": "NN"}]

# Sentiment polarity
nlp.polarity("ខ្ញុំចូលចិត្តប្រទេសខ្មែរណាស់")
# → {"label": "positive", "confidence": 0.9944, "scores": {"negative": 0.0023, "neutral": 0.0032, "positive": 0.9944}}

# All tasks at once
nlp.analyze("គាត់ចូលចិត្តអានសៀវភៅ")
```

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
| `segment(text)` | `list[str]` | Segmented word list |
| `pos(text)` | `list[dict]` | Word + POS label |
| `polarity(text)` | `dict` | Sentiment polarity |
| `analyze(text)` | `dict` | All tasks combined |

Raw token-level outputs: `seg_tokens()`, `pos_tokens()`.

## License

MIT
