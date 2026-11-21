# inference/cpu_runner.py
import os
import gc
import sys
import platform
import argparse
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from config import CyberCoreConfig
from model.architecture import CyberCoreZero

def configure_cpu_backend():
    arch = platform.machine().lower()
    if "arm" in arch or "aarch64" in arch:
        torch.backends.quantized.engine = "qnnpack"
    else:
        torch.backends.quantized.engine = "fbgemm"
    threads = max(1, os.cpu_count() or 4)
    torch.set_num_threads(threads)
    return threads, torch.backends.quantized.engine

class CPURunner:
    def __init__(
        self,
        ckpt_path="checkpoints/cybercore_1.5b_inference.pt",
        tok_path="tokenizer/out/tokenizer.json",
        quantize_int8=True
    ):
        threads, engine = configure_cpu_backend()
        print(f"[*] تشغيل CyberCore-Zero 1.5B على CPU | الخيوط: {threads} | محرك الضغط: {engine}")

        self.tokenizer = Tokenizer.from_file(tok_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        cfg = CyberCoreConfig(**ckpt["config"])
        self.model = CyberCoreZero(cfg, gradient_checkpointing=False)
        float_state = {k: v.float() for k, v in ckpt["model"].items()}
        self.model.load_state_dict(float_state, strict=False)

        if cfg.tie_word_embeddings:
            self.model.lm_head.weight = self.model.embed_tokens.weight

        del ckpt, float_state
        gc.collect()
        self.model.eval()

        if quantize_int8:
            print("[*] جارٍ ضغط طبقات 1.5B إلى Dynamic INT8 لتقليص استهلاك الرام وتسريع التوليد...")
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            gc.collect()
        print("[+] النموذج 1.5B جاهز بالكامل على CPU.")

    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, stream: bool = False) -> str:
        ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([ids], dtype=torch.long)
        generated_ids = []
        printed_text = ""

        for token_id in self.model.generate_stream(input_tensor, max_new_tokens=max_new_tokens, temperature=temperature):
            generated_ids.append(token_id)
            if stream:
                full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                # تجنب طباعة رموز UTF-8 غير مكتملة أثناء بث العربية
                if not full_text.endswith("\ufffd"):
                    delta = full_text[len(printed_text):]
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        printed_text = full_text

        final_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if stream:
            delta = final_text[len(printed_text):]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
            print()
        return final_text

def interactive_chat(ckpt_path, tok_path, no_int8=False):
    runner = CPURunner(ckpt_path=ckpt_path, tok_path=tok_path, quantize_int8=not no_int8)
    print("\n=== CyberCore-Zero 1.5B CPU Terminal (اكتب exit للخروج أو /env لتغيير البيئة) ===")
    env = input("اختر البيئة الافتراضية [ish / segfault / kali / ubuntu] (افتراضي kali): ").strip() or "kali"

    while True:
        user_msg = input(f"\n[{env}] أنت> ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break
        if user_msg.startswith("/env "):
            env = user_msg.split(" ", 1)[1].strip()
            print(f"[*] تم التبديل إلى بيئة: {env}")
            continue
        prompt = (
            f"<|bos|><|sys|>أنت CyberCore-Zero 1.5B خبير الأمن السيبراني والبايثون لبيئة {env}.<|eos|>\n"
            f"<|usr|><|env_{env}|> {user_msg}<|eos|>\n<|ast|>"
        )
        print("\n[CyberCore-Zero 1.5B]> ", end="", flush=True)
        runner.generate_text(prompt, max_new_tokens=768, temperature=0.2, stream=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/cybercore_1.5b_inference.pt")
    parser.add_argument("--tok", type=str, default="tokenizer/out/tokenizer.json")
    parser.add_argument("--no_int8", action="store_true")
    args = parser.parse_args()
    interactive_chat(args.ckpt, args.tok, args.no_int8)
