# model/architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import CyberCoreConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(norm + self.eps) * self.weight.float()).type_as(x)

def precompute_rope_cache(seq_len: int, head_dim: int, theta: float = 500000.0, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: CyberCoreConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_kv=None, use_cache: bool = False):
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        start_pos = past_kv[0].shape[2] if past_kv is not None else 0
        q, k = apply_rope(q, k, cos[start_pos : start_pos + seqlen], sin[start_pos : start_pos + seqlen])

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None
        k_rep = k.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else k
        v_rep = v.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else v

        if past_kv is None and seqlen > 1:
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        elif seqlen == 1:
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
        else:
            total_len = k_rep.shape[2]
            mask = torch.ones((seqlen, total_len), dtype=torch.bool, device=x.device).tril(diagonal=total_len - seqlen)
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(out), new_kv

class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: CyberCoreConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: CyberCoreConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.mlp = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_kv=None, use_cache: bool = False):
        attn_out, new_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv

class CyberCoreZero(nn.Module):
    def __init__(self, cfg: CyberCoreConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = gradient_checkpointing
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _ensure_rope(self, needed_len: int, device):
        if self.rope_cos is None or self.rope_cos.shape[0] < needed_len:
            cos, sin = precompute_rope_cache(
                max(needed_len, self.cfg.max_seq_len),
                self.cfg.dim // self.cfg.n_heads,
                self.cfg.rope_theta,
                device=device
            )
            self.rope_cos, self.rope_sin = cos, sin

    def _chunked_cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        """حساب الخسارة على شرائح صغيرة لمنع انفجار ذاكرة VRAM أثناء تدريب 1.5B على Colab"""
        hidden_flat = hidden.reshape(-1, hidden.size(-1))
        targets_flat = targets.reshape(-1)
        valid_tokens = (targets_flat != self.cfg.pad_token_id).sum().clamp(min=1)

        def loss_fn(h_chunk, t_chunk):
            logits_chunk = self.lm_head(h_chunk).float()
            return F.cross_entropy(logits_chunk, t_chunk, ignore_index=self.cfg.pad_token_id, reduction="sum")

        total_loss = hidden_flat.new_zeros((), dtype=torch.float32)
        for i in range(0, hidden_flat.size(0), chunk_size):
            h_c = hidden_flat[i : i + chunk_size]
            t_c = targets_flat[i : i + chunk_size]
            if self.training and self.gradient_checkpointing:
                total_loss = total_loss + checkpoint(loss_fn, h_c, t_c, use_reentrant=False)
            else:
                total_loss = total_loss + loss_fn(h_c, t_c)

        return total_loss / valid_tokens

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, past_kvs=None, use_cache: bool = False):
        bsz, seqlen = input_ids.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        self._ensure_rope(past_len + seqlen, input_ids.device)
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)

        new_kvs = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            if self.training and self.gradient_checkpointing and not use_cache:
                def custom_forward(hidden_states):
                    out, _ = layer(hidden_states, cos, sin, None, False)
                    return out
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x, kv = layer(x, cos, sin, past_kv=layer_past, use_cache=use_cache)
                if use_cache:
                    new_kvs.append(kv)

        x = self.norm(x)

        if targets is not None:
            loss = self._chunked_cross_entropy(x, targets)
            return None, loss

        logits = self.lm_head(x)
        return (logits, None, new_kvs) if use_cache else (logits, None)

    @torch.inference_mode()
    def generate_stream(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.9):
        """توليد متدفق (Token-by-Token) لتجربة سريعة ومباشرة على CPU"""
        past_kvs = None
        next_input = input_ids
        eos_id = self.cfg.eos_token_id

        for _ in range(max_new_tokens):
            logits, _, past_kvs = self(next_input, past_kvs=past_kvs, use_cache=True)
            next_logits = logits[:, -1, :].float()

            if temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                mask = cum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = float("-inf")
                sampled = torch.multinomial(F.softmax(sorted_logits, dim=-1), num_samples=1)
                next_token = sorted_indices.gather(-1, sampled)

            token_id = next_token.item()
            if token_id == eos_id:
                break
            yield token_id
            next_input = next_token

    def count_parameters(self) -> int:
        # حساب المعاملات الفريدة فقط عند تفعيل Tied Embeddings
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())
