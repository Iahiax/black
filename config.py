# config.py
from dataclasses import dataclass

@dataclass
class CyberCoreConfig:
    # الإعداد الافتراضي: ~1.5 مليار معامل (يمكن تغييره لـ 7B برفع dim=4096 و n_layers=32)
    vocab_size: int = 65536
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4             # Grouped-Query Attention (GQA) لتقليل الذاكرة 4 أضعاف
    intermediate_size: int = 5632   # SwiGLU FFN
    max_seq_len: int = 16384        # نافذة سياقية طويلة لقراءة السكربتات والسجلات
    rope_theta: float = 500000.0    # دعم السياق الطويل (RoPE scaling)
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0            # 0 في التدريب المسبق الحديث لثبات أعلى

    # الرموز الخاصة بالوكيل والبيئات
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
