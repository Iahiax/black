# inference/cpu_runner.py
import os
import argparse
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from config import CyberCoreConfig
from model.architecture import CyberCoreZero

class CPURunner:
    def __init__(self, ckpt_path="checkpoints/cybercore_inference.pt", tok_path="tokenizer/out/tokenizer.json", quantize_int8=True):
        # استغلال جميع أنوية المعالج المركزي (CPU)
        num_threads = max(1, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"[*] تهيئة محرك CPU النقي (PyTorch) باستخدام {num_threads} خيوط معالجة...")

        self.tokenizer = Tokenizer.from_file(tok_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        cfg = CyberCoreConfig(**ckpt["config"])
        self.model = CyberCoreZero(cfg, gradient_checkpointing=False)
        self.model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
        self.model.eval()

        if quantize_int8:
            print("[*] تطبيق ضغط PyTorch Dynamic INT8 لتقليل استهلاك RAM وتسريع الـ CPU...")
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
        print(f"[+] النموذج جاهز للتشغيل على CPU ({cfg.profile}).")

    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([ids], dtype=torch.long)
        out_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=self.model.cfg.eos_token_id
        )[0].tolist()
        new_tokens = out_ids[len(ids):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

def interactive_chat(ckpt_path, tok_path, no_int8=False):
    runner = CPURunner(ckpt_path=ckpt_path, tok_path=tok_path, quantize_int8=not no_int8)
    print("\n=== CyberCore-Zero CPU Shell (اكتب exit للخروج) ===")
    print("البيئات المدعومة: ish | segfault | kali | ubuntu")
    env = input("اختر البيئة الافتراضية [kali]: ").strip() or "kali"

    while True:
        user_msg = input(f"\n[{env}] أنت> ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break
        prompt = (
            f"<|bos|><|sys|>أنت CyberCore-Zero خبير الأمن السيبراني والبايثون لبيئة {env}.<|eos|>\n"
            f"<|usr|><|env_{env}|> {user_msg}<|eos|>\n<|ast|>"
        )
        reply = runner.generate_text(prompt)
        print(f"\n[CyberCore-Zero]>\n{reply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/cybercore_inference.pt")
    parser.add_argument("--tok", type=str, default="tokenizer/out/tokenizer.json")
    parser.add_argument("--no_int8", action="store_true")
    args = parser.parse_args()
    interactive_chat(args.ckpt, args.tok, args.no_int8)
