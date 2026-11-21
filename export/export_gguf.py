# export/export_gguf.py
import os
import json
import shutil
import subprocess
import torch
from safetensors.torch import save_file
from config import CyberCoreConfig

def export_and_convert_to_gguf(
    ckpt_path="checkpoints/cybercore_final.pt",
    tokenizer_json="tokenizer/out/tokenizer.json",
    hf_out_dir="export/hf_model",
    llama_cpp_dir="./llama.cpp",
    quant_type="Q4_K_M"
):
    os.makedirs(hf_out_dir, exist_ok=True)
    cfg = CyberCoreConfig()

    print("[1/4] تحويل أوزان النموذج المولود من الصفر إلى Safetensors متوافق مع معمارية Llama...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    mapped = {}
    for k, v in state_dict.items():
        if k == "lm_head.weight":
            mapped["lm_head.weight"] = v.to(torch.float16).contiguous()
        else:
            mapped[f"model.{k}"] = v.to(torch.float16).contiguous()

    save_file(mapped, os.path.join(hf_out_dir, "model.safetensors"))

    # كتابة ملف الإعدادات القياسي الذي يقرأه llama.cpp
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": False,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "torch_dtype": "float16",
    }
    with open(os.path.join(hf_out_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    shutil.copy(tokenizer_json, os.path.join(hf_out_dir, "tokenizer.json"))
    with open(os.path.join(hf_out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|bos|>", "eos_token": "<|eos|>"}, f)

    fp16_gguf = "export/cybercore-zero-f16.gguf"
    quant_gguf = f"export/cybercore-zero-{quant_type.lower()}.gguf"

    print("[2/4] تحويل النموذج إلى GGUF خام (FP16)...")
    subprocess.run([
        "python3", os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
        hf_out_dir, "--outfile", fp16_gguf, "--outtype", "f16"
    ], check=True)

    print(f"[3/4] ضغط النموذج إلى {quant_type} لتشغيل فائق السرعة...")
    subprocess.run([
        os.path.join(llama_cpp_dir, "build/bin/llama-quantize"),
        fp16_gguf, quant_gguf, quant_type
    ], check=True)

    print(f"[4/4] اكتمل البناء! ملفك العالمي الجاهز: {quant_gguf}")

if __name__ == "__main__":
    export_and_convert_to_gguf()
