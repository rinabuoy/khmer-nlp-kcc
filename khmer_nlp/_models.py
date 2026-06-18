from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    seq_len = xq.shape[2]
    freqs = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(0)
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_c * freqs).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(3).type_as(xk)
    return xq_out, xk_out


class SwiGLU(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(hid_dim, pf_dim, bias=False)
        self.w2 = nn.Linear(pf_dim, hid_dim, bias=False)
        self.w3 = nn.Linear(hid_dim, pf_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim, bias=False)
        self.fc_k = nn.Linear(hid_dim, hid_dim, bias=False)
        self.fc_v = nn.Linear(hid_dim, hid_dim, bias=False)
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias=False)
        self.dropout_p = dropout

    def forward(self, query, key, value, mask=None, freqs_cis=None):
        B, Lq = query.shape[:2]
        Q = self.fc_q(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.fc_k(key).view(B, key.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        V = self.fc_v(value).view(B, key.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        if freqs_cis is not None:
            Q, K = apply_rotary_emb(Q, K, freqs_cis)
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_mask = torch.zeros(mask.shape, dtype=query.dtype, device=query.device)
                attn_mask = attn_mask.masked_fill(~mask, float('-inf'))
            else:
                attn_mask = torch.zeros_like(mask, dtype=query.dtype)
                attn_mask = attn_mask.masked_fill(mask == 0, float('-inf'))
        x = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.fc_o(x.transpose(1, 2).contiguous().view(B, Lq, -1))


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float):
        super().__init__()
        self.self_attn_norm = RMSNorm(hid_dim)
        self.ff_norm = RMSNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.ffn = SwiGLU(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, freqs_cis=None):
        normed = self.self_attn_norm(x)
        x = x + self.dropout(self.self_attention(normed, normed, normed, mask, freqs_cis))
        return x + self.dropout(self.ffn(self.ff_norm(x)))


class Decoder(nn.Module):
    def __init__(self, output_dim: int, hid_dim: int, n_layers: int, n_heads: int,
                 pf_dim: int, dropout: float, device, max_seq_len: int = 4096):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        head_dim = hid_dim // n_heads
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, max_seq_len))
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, trg_mask):
        trg = self.dropout(self.tok_embedding(trg))
        for layer in self.layers:
            trg = layer(trg, trg_mask, self.freqs_cis)
        return self.fc_out(self.norm(trg))


class KhmerLM(nn.Module):
    def __init__(self, decoder: Decoder, trg_pad_idx: int = 0):
        super().__init__()
        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(
            torch.ones((trg.shape[1], trg.shape[1]), device=trg.device)
        ).bool()
        return pad_mask & sub_mask

    def forward(self, trg: torch.Tensor) -> torch.Tensor:
        return self.decoder(trg, self.make_trg_mask(trg))


class ClassificationHead(nn.Module):
    def __init__(self, hid_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hid_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.dropout(F.gelu(self.norm(self.dense(x)))))


class TransAdapterLM(nn.Module):
    def __init__(self, hid_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hid_dim, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(self.norm(hidden)))


class KhmerMultiTaskClassifier(nn.Module):
    def __init__(self, backbone: KhmerLM, num_pos_labels: int, num_seg_labels: int,
                 num_ner_labels: int, num_nova_word_labels: int, num_nova_inner_labels: int,
                 num_profanity_labels: int, num_polarity_labels: int):
        super().__init__()
        self.backbone = backbone
        hid_dim = backbone.decoder.norm.weight.shape[0]
        vocab_size = backbone.decoder.fc_out.out_features
        self.pos_classifier        = ClassificationHead(hid_dim, num_pos_labels)
        self.seg_classifier        = ClassificationHead(hid_dim, num_seg_labels)
        self.ner_classifier        = ClassificationHead(hid_dim, num_ner_labels)
        self.nova_word_classifier  = ClassificationHead(hid_dim, num_nova_word_labels)
        self.nova_inner_classifier = ClassificationHead(hid_dim, num_nova_inner_labels)
        self.profanity_classifier  = ClassificationHead(hid_dim, num_profanity_labels)
        self.polarity_classifier   = ClassificationHead(hid_dim, num_polarity_labels)
        self.km2en_adapter         = TransAdapterLM(hid_dim, vocab_size)
        self.en2km_adapter         = TransAdapterLM(hid_dim, vocab_size)
        self.pad_idx               = backbone.trg_pad_idx

    def make_full_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = self.make_full_mask(input_ids)
        dec = self.backbone.decoder
        x = dec.dropout(dec.tok_embedding(input_ids))
        for layer in dec.layers:
            x = layer(x, mask, dec.freqs_cis)
        return dec.norm(x)

    def _encode_causal(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = self.backbone.make_trg_mask(input_ids)
        dec = self.backbone.decoder
        x = dec.dropout(dec.tok_embedding(input_ids))
        for layer in dec.layers:
            x = layer(x, mask, dec.freqs_cis)
        return dec.norm(x)

    def _encode_pooled(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._encode(input_ids)
        mask = (input_ids != self.pad_idx).unsqueeze(-1).float()
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids: torch.Tensor, task: str) -> torch.Tensor:
        if task in ("km2en", "en2km"):
            x = self._encode_causal(input_ids)
            return self.km2en_adapter(x) if task == "km2en" else self.en2km_adapter(x)
        if task in ("profanity", "polarity"):
            x = self._encode_pooled(input_ids)
            return self.profanity_classifier(x) if task == "profanity" else self.polarity_classifier(x)
        x = self._encode(input_ids)
        if task == "pos":        return self.pos_classifier(x)
        if task == "seg":        return self.seg_classifier(x)
        if task == "ner":        return self.ner_classifier(x)
        if task == "nova_word":  return self.nova_word_classifier(x)
        if task == "nova_inner": return self.nova_inner_classifier(x)
        return self.nova_inner_classifier(x)
