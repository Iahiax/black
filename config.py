# config.py
from dataclasses import dataclass, asdict

@dataclass
class CyberCoreConfig:
    profile: str = "colab_t4"
    vocab_size: int = 65536
    dim: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    intermediate_size: int = 2816
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def to_dict(self):
        return asdict(self)

def get_config(profile: str = "colab_t4") -> CyberCoreConfig:
    profiles = {
        # مناسب لـ Google Colab المجاني (T4 15GB VRAM) ~ 330M معامل
        "colab_t4": CyberCoreConfig(
            profile="colab_t4", dim=1024, n_layers=16, n_heads=16,
            n_kv_heads=4, intermediate_size=2816, max_seq_len=4096
        ),
        # مناسب لـ Google Colab Pro (A100 40GB) ~ 1.5B معامل
        "colab_a100": CyberCoreConfig(
            profile="colab_a100", dim=2048, n_layers=24, n_heads=16,
            n_kv_heads=4, intermediate_size=5632, max_seq_len=8192
        ),
        # النموذج العالمي الثقيل (7B معامل) - يحتاج A100 80GB أو Multi-GPU
        "heavy_7b": CyberCoreConfig(
            profile="heavy_7b", dim=4096, n_layers=32, n_heads=32,
            n_kv_heads=8, intermediate_size=14336, max_seq_len=16384
        ),
    }
    if profile not in profiles:
        raise ValueError(f"وضع غير معروف: {profile}. المتاح: {list(profiles.keys())}")
    return profiles[profile]
