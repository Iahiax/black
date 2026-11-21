# config.py
from dataclasses import dataclass, asdict

@dataclass
class CyberCoreConfig:
    model_name: str = "CyberCore-Zero-1.5B"
    vocab_size: int = 65536
    dim: int = 2048
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 4             # Grouped-Query Attention (يقلل ذاكرة KV-Cache على CPU بنسبة 75%)
    intermediate_size: int = 6144   # SwiGLU FFN
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True  # ربط طبقة الإدخال والإخراج لزيادة دقة 1.5B وتقليل VRAM
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def to_dict(self):
        return asdict(self)
