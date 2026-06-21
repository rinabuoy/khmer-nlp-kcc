from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocab ──────────────────────────────────────────────────────────────────────

_LATIN = list("abcdefghijklmnopqrstuvwxyz")
_PHONES = [
    '.', 'a', 'aa', 'ae', 'ao', 'aə', 'c', 'ch', 'e', 'ea', 'ee', 'f', 'g',
    'h', 'i', 'ie', 'ii', 'iə', 'j', 'k', 'kh', 'l', 'm', 'n', 'o', 'oa',
    'oo', 'p', 'ph', 'r', 's', 't', 'th', 'u', 'uu', 'uə', 'w', 'z', 'ŋ',
    'ɑ', 'ɑɑ', 'ɓ', 'ɔ', 'ɔɔ', 'ɗ', 'ə', 'əə', 'ɛ', 'ɛə', 'ɛɛ', 'ɨ', 'ɨə',
    'ɨɨ', 'ɲ', 'ʔ',
]
for _p in _PHONES:
    if _p not in _LATIN:
        _LATIN.append(_p)

_CHARS = (
    ['_PAD', 'UNK', 'SOS', 'EOS', '#']
    + list('កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ')
    + list('ាិីឹឺុូួើឿៀេែៃោៅំះៈ')
    + list('្')
    + list('៉៊់៌៍៎៏័')
    + _LATIN
    + list('០១២៣៤៥៦៧៨៩0123456789')
    + [' ']
)

_C2I = {c: i for i, c in enumerate(_CHARS)}
_I2C = {i: c for i, c in enumerate(_CHARS)}

_INPUT_DIM = len(_CHARS)
_HID_DIM = 192
_N_LAYERS = 3
_N_HEADS = 8
_PF_DIM = 192 * 4
_DROPOUT = 0.1


# ── Model layers ───────────────────────────────────────────────────────────────

class _MHA(nn.Module):
    def __init__(self, hid, heads, drop, device):
        super().__init__()
        self.heads = heads
        self.head_dim = hid // heads
        self.fc_q = nn.Linear(hid, hid)
        self.fc_k = nn.Linear(hid, hid)
        self.fc_v = nn.Linear(hid, hid)
        self.fc_o = nn.Linear(hid, hid)
        self.drop = nn.Dropout(drop)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        Q = self.fc_q(q).view(bs, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(k).view(bs, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(v).view(bs, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        e = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e10)
        a = torch.softmax(e, dim=-1)
        x = torch.matmul(self.drop(a), V).permute(0, 2, 1, 3).contiguous()
        x = x.view(bs, -1, self.heads * self.head_dim)
        return self.fc_o(x), a


class _FF(nn.Module):
    def __init__(self, hid, pf, drop):
        super().__init__()
        self.fc1 = nn.Linear(hid, pf)
        self.fc2 = nn.Linear(pf, hid)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


class _EncLayer(nn.Module):
    def __init__(self, hid, heads, pf, drop, device):
        super().__init__()
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.attn = _MHA(hid, heads, drop, device)
        self.ff = _FF(hid, pf, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, src, mask):
        _s, _ = self.attn(src, src, src, mask)
        src = self.norm1(src + self.drop(_s))
        return self.norm2(src + self.drop(self.ff(src)))


class _DecLayer(nn.Module):
    def __init__(self, hid, heads, pf, drop, device):
        super().__init__()
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.norm3 = nn.LayerNorm(hid)
        self.self_attn = _MHA(hid, heads, drop, device)
        self.enc_attn = _MHA(hid, heads, drop, device)
        self.ff = _FF(hid, pf, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, trg, enc, trg_mask, src_mask):
        _t, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.drop(_t))
        _t, attn = self.enc_attn(trg, enc, enc, src_mask)
        trg = self.norm2(trg + self.drop(_t))
        return self.norm3(trg + self.drop(self.ff(trg))), attn


class _Encoder(nn.Module):
    def __init__(self, vocab, hid, layers, heads, pf, drop, device, max_len=100):
        super().__init__()
        self.device = device
        self.tok_emb = nn.Embedding(vocab, hid)
        self.pos_emb = nn.Embedding(max_len, hid)
        self.layers = nn.ModuleList([_EncLayer(hid, heads, pf, drop, device) for _ in range(layers)])
        self.drop = nn.Dropout(drop)
        self.scale = torch.sqrt(torch.FloatTensor([hid])).to(device)

    def forward(self, src, mask):
        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)
        src = self.drop(self.tok_emb(src) * self.scale + self.pos_emb(pos))
        for layer in self.layers:
            src = layer(src, mask)
        return src


class _Decoder(nn.Module):
    def __init__(self, vocab, hid, layers, heads, pf, drop, device, max_len=100):
        super().__init__()
        self.device = device
        self.tok_emb = nn.Embedding(vocab, hid)
        self.pos_emb = nn.Embedding(max_len, hid)
        self.layers = nn.ModuleList([_DecLayer(hid, heads, pf, drop, device) for _ in range(layers)])
        self.fc_out = nn.Linear(hid, vocab)
        self.drop = nn.Dropout(drop)
        self.scale = torch.sqrt(torch.FloatTensor([hid])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)
        trg = self.drop(self.tok_emb(trg) * self.scale + self.pos_emb(pos))
        for layer in self.layers:
            trg, attn = layer(trg, enc_src, trg_mask, src_mask)
        return self.fc_out(trg), attn


class _G2PModel(nn.Module):
    def __init__(self, enc, dec, pad_idx, device):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.pad_idx = pad_idx
        self.device = device

    def _src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _trg_mask(self, trg):
        pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(torch.ones((trg.shape[1], trg.shape[1]), device=self.device)).bool()
        return pad_mask & sub_mask

    def forward(self, src, trg):
        enc_src = self.encoder(src, self._src_mask(src))
        return self.decoder(trg, enc_src, self._trg_mask(trg), self._src_mask(src))


# ── Public helpers ─────────────────────────────────────────────────────────────

def load_g2p(path: Optional[str] = None, device: Optional[torch.device] = None) -> _G2PModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path is None:
        path = str(Path(__file__).parent.parent / "g2p_final_trans.pt")
    enc = _Encoder(_INPUT_DIM, _HID_DIM, _N_LAYERS, _N_HEADS, _PF_DIM, _DROPOUT, device)
    dec = _Decoder(_INPUT_DIM, _HID_DIM, _N_LAYERS, _N_HEADS, _PF_DIM, _DROPOUT, device)
    model = _G2PModel(enc, dec, pad_idx=0, device=device).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def g2p_word(word: str, model: _G2PModel, device: torch.device, max_len: int = 100) -> list[str]:
    src_ids = (
        [_C2I['SOS']]
        + [_C2I.get(c, _C2I['UNK']) for c in word]
        + [_C2I['EOS']]
    )
    src = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    src_mask = model._src_mask(src)
    enc_src = model.encoder(src, src_mask)

    trg_ids = [_C2I['SOS']]
    for _ in range(max_len):
        trg = torch.LongTensor(trg_ids).unsqueeze(0).to(device)
        trg_mask = model._trg_mask(trg)
        out, _ = model.decoder(trg, enc_src, trg_mask, src_mask)
        pred = out.argmax(2)[:, -1].item()
        trg_ids.append(pred)
        if pred == _C2I['EOS']:
            break

    tokens = [_I2C[i] for i in trg_ids[1:-1]]
    return [t for t in ''.join(tokens).split(' ') if t]


def phoneme_cer(phones1: list[str], phones2: list[str]) -> float:
    """Edit distance at phoneme-token level divided by len(phones1)."""
    r, h = phones1, phones2
    m, n = len(r), len(h)
    if m == 0:
        return float(n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if r[i - 1] == h[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return round(dp[n] / m, 4)
