# inference/cpu_runner.py
import os
import re
import gc
import sys
import platform
import argparse
import numpy as np
import torch
import torch.nn as nn
import gguf
from tokenizers import Tokenizer
from config import CyberCoreConfig
from model.architecture import CyberCoreZero
from inference.search_engine import LiveSearchEngine


def configure_cpu_backend():
    arch = platform.machine().lower()
    torch.backends.quantized.engine = "qnnpack" if ("arm" in arch or "aarch64" in arch) else "fbgemm"
    threads = max(1, os.cpu_count() or 4)
    torch.set_num_threads(threads)
    return threads, torch.backends.quantized.engine


def _read_gguf_kv(reader: gguf.GGUFReader, key: str, default=None):
    field = reader.get_field(key)
    if field is None:
        return default
    if hasattr(field, "contents"):
        return field.contents()
    val = field.parts[field.data[0]]
    if field.types[0] == gguf.GGUFValueType.STRING:
        return bytes(val).decode("utf-8")
    return val.item()


def unpermute_from_gguf(w: np.ndarray, n_heads: int) -> np.ndarray:
    out_dim, in_dim = w.shape
    head_dim = out_dim // n_heads
    return (
        w.reshape(n_heads, head_dim // 2, 2, in_dim)
        .transpose(0, 2, 1, 3)
        .reshape(out_dim, in_dim)
    )


def map_gguf_to_pt_key(name: str) -> str | None:
    if name == "token_embd.weight":
        return "embed_tokens.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name.startswith("blk."):
        parts = name.split(".")
        idx = parts[1]
        sub = ".".join(parts[2:])
        mapping = {
            "attn_norm.weight": f"layers.{idx}.input_layernorm.weight",
            "attn_q.weight": f"layers.{idx}.self_attn.q_proj.weight",
            "attn_k.weight": f"layers.{idx}.self_attn.k_proj.weight",
            "attn_v.weight": f"layers.{idx}.self_attn.v_proj.weight",
            "attn_output.weight": f"layers.{idx}.self_attn.o_proj.weight",
            "ffn_norm.weight": f"layers.{idx}.post_attention_layernorm.weight",
            "ffn_gate.weight": f"layers.{idx}.mlp.gate_proj.weight",
            "ffn_up.weight": f"layers.{idx}.mlp.up_proj.weight",
            "ffn_down.weight": f"layers.{idx}.mlp.down_proj.weight",
        }
        return mapping.get(sub)
    return None


class CPURunner:
    def __init__(self, gguf_path="export/cybercore-1.5b-q8_0.gguf", tok_path=None, quantize_int8=True):
        threads, engine = configure_cpu_backend()
        print(f"[*] تحميل ملف GGUF مباشرة على CPU | الخيوط: {threads} | المحرك: {engine}")

        self.search_engine = LiveSearchEngine()
        reader = gguf.GGUFReader(gguf_path)

        cfg = CyberCoreConfig(
            model_name=str(_read_gguf_kv(reader, "general.name", "CyberCore-Zero-1.5B")),
            vocab_size=int(_read_gguf_kv(reader, "llama.vocab_size", 65536)),
            dim=int(_read_gguf_kv(reader, "llama.embedding_length", 2048)),
            n_layers=int(_read_gguf_kv(reader, "llama.block_count", 28)),
            n_heads=int(_read_gguf_kv(reader, "llama.attention.head_count", 16)),
            n_kv_heads=int(_read_gguf_kv(reader, "llama.attention.head_count_kv", 4)),
            intermediate_size=int(_read_gguf_kv(reader, "llama.feed_forward_length", 6144)),
            max_seq_len=int(_read_gguf_kv(reader, "llama.context_length", 8192)),
            rope_theta=float(_read_gguf_kv(reader, "llama.rope.freq_base", 500000.0)),
            rms_norm_eps=float(_read_gguf_kv(reader, "llama.attention.layer_norm_rms_epsilon", 1e-5)),
            tie_word_embeddings=bool(_read_gguf_kv(reader, "cybercore.tie_word_embeddings", True)),
            bos_token_id=int(_read_gguf_kv(reader, "tokenizer.ggml.bos_token_id", 1)),
            eos_token_id=int(_read_gguf_kv(reader, "tokenizer.ggml.eos_token_id", 2)),
            pad_token_id=int(_read_gguf_kv(reader, "tokenizer.ggml.pad_token_id", 0)),
        )

        embedded_tok = _read_gguf_kv(reader, "tokenizer.huggingface.json")
        if embedded_tok:
            self.tokenizer = Tokenizer.from_str(embedded_tok)
        elif tok_path and os.path.exists(tok_path):
            self.tokenizer = Tokenizer.from_file(tok_path)
        else:
            raise FileNotFoundError("لم يُعثر على المُرمّز داخل GGUF؛ مرر --tok لمسار tokenizer.json.")

        state_dict = {}
        for t in reader.tensors:
            pt_key = map_gguf_to_pt_key(t.name)
            if pt_key is None:
                continue

            oshape = tuple(int(d) for d in reversed(t.shape))
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                arr = np.array(t.data, dtype=np.float32, copy=True).reshape(oshape)
            elif t.tensor_type == gguf.GGMLQuantizationType.F16:
                arr = np.array(t.data, dtype=np.float16).astype(np.float32).reshape(oshape)
            else:
                arr = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(oshape)

            if t.name.endswith(".attn_q.weight"):
                arr = unpermute_from_gguf(arr, cfg.n_heads)
            elif t.name.endswith(".attn_k.weight"):
                arr = unpermute_from_gguf(arr, cfg.n_kv_heads)

            state_dict[pt_key] = torch.from_numpy(np.ascontiguousarray(arr))

        del reader
        gc.collect()

        self.model = CyberCoreZero(cfg, gradient_checkpointing=False)
        self.model.load_state_dict(state_dict, strict=False)
        if cfg.tie_word_embeddings or "lm_head.weight" not in state_dict:
            self.model.lm_head.weight = self.model.embed_tokens.weight

        del state_dict
        gc.collect()
        self.model.eval()

        if quantize_int8:
            print("[*] تطبيق ضغط PyTorch Dynamic INT8 للتسريع على CPU ...")
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            gc.collect()

        tor_status = "متصل (9050)" if self.search_engine.is_tor_running() else "غير متصل"
        print(f"[+] تم تحميل {cfg.model_name} (GGUF) بنجاح | حالة Tor: {tor_status}")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        stream: bool = False,
        keep_special_tokens: bool = False,
    ) -> str:
        ids = self.tokenizer.encode(prompt).ids
        # قص البداية إذا تجاوز السياق المسموح
        max_ctx = self.model.cfg.max_seq_len - max_new_tokens
        if len(ids) > max_ctx and max_ctx > 0:
            ids = ids[-max_ctx:]

        input_tensor = torch.tensor([ids], dtype=torch.long)
        generated_ids = []
        printed_text = ""

        for token_id in self.model.generate_stream(input_tensor, max_new_tokens=max_new_tokens, temperature=temperature):
            generated_ids.append(token_id)
            if stream:
                full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=not keep_special_tokens)
                if not full_text.endswith("\ufffd"):
                    delta = full_text[len(printed_text):]
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        printed_text = full_text

        final_text = self.tokenizer.decode(generated_ids, skip_special_tokens=not keep_special_tokens)
        if stream:
            delta = final_text[len(printed_text):]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
            print()
        return final_text

    def generate_with_live_search(
        self,
        prompt: str,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        max_search_rounds: int = 3,
        stream: bool = True,
    ) -> str:
        """يسمح لنموذج GGUF بطلب البحث في الإنترنت العادي أو المظلم تلقائياً ثم إكمال الإجابة"""
        current_prompt = prompt
        final_reply = ""

        for round_idx in range(max_search_rounds + 1):
            raw_reply = self.generate_text(
                current_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=stream,
                keep_special_tokens=True,
            )

            web_match = re.search(r"<\|web_search\|>\s*([^\n<]+)", raw_reply)
            tor_match = re.search(r"<\|tor_search\|>\s*([^\n<]+)", raw_reply)

            if round_idx < max_search_rounds and (web_match or tor_match):
                if tor_match:
                    q = tor_match.group(1).strip()
                    print(f"\n[*] [GGUF -> Tor Search] جارٍ البحث في الشبكة المظلمة عن: {q} ...")
                    obs = self.search_engine.search_dark_web(q, max_results=5, fetch_top_onion=True)
                else:
                    q = web_match.group(1).strip()
                    print(f"\n[*] [GGUF -> Web Search] جارٍ البحث في الإنترنت العادي عن: {q} ...")
                    obs = self.search_engine.search_surface_web(q, max_results=5)

                print(f"[+] تم جلب البيانات الحية ({len(obs)} حرف) وحقنها في سياق GGUF...\n")
                current_prompt += f"{raw_reply}<|eos|>\n<|obs|>\n{obs[:2500]}\n<|eos|>\n<|ast|>"
                if stream:
                    print("[CyberCore-Zero 1.5B]> ", end="", flush=True)
                continue

            final_reply = re.sub(r"<\|[^|]+\|>", "", raw_reply).strip()
            break

        return final_reply


def interactive_chat(gguf_path, tok_path=None, no_int8=False):
    runner = CPURunner(gguf_path=gguf_path, tok_path=tok_path, quantize_int8=not no_int8)
    print("\n=== CyberCore-Zero 1.5B GGUF CPU Shell ===")
    print("الأوامر المتاحة:")
    print("  /web <استعلام أو رابط>     -> بحث حي في الإنترنت العادي ثم التحليل بالنموذج")
    print("  /tor <استعلام أو .onion>   -> بحث حي في الشبكة المظلمة عبر Tor ثم التحليل بالنموذج")
    print("  /tor_ip                    -> فحص اتصال Tor وتجديد الهوية")
    print("  /env <ish|segfault|kali|ubuntu> -> تبديل البيئة")
    print("  exit                       -> خروج")

    env = input("\nاختر البيئة الافتراضية [kali]: ").strip() or "kali"

    while True:
        user_msg = input(f"\n[{env}] أنت> ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break

        if user_msg.startswith("/env "):
            env = user_msg.split(" ", 1)[1].strip()
            print(f"[*] تم التبديل إلى بيئة: {env}")
            continue

        if user_msg == "/tor_ip":
            print("[*]", runner.search_engine.check_tor_ip())
            print("[*]", runner.search_engine.renew_tor_identity())
            continue

        injected_obs = ""
        clean_task = user_msg

        if user_msg.startswith("/web "):
            query = user_msg[5:].strip()
            print(f"[*] جارٍ البحث في الإنترنت العادي عن: {query} ...")
            injected_obs = runner.search_engine.search_surface_web(query, max_results=5)
            clean_task = f"بناءً على نتائج البحث التالية حول ({query})، قدم تحليلاً تقنياً أو كود بايثون عملي:"
        elif user_msg.startswith("/tor "):
            query = user_msg[5:].strip()
            print(f"[*] جارٍ البحث في الشبكة المظلمة (Tor) عن: {query} ...")
            injected_obs = runner.search_engine.search_dark_web(query, max_results=5, fetch_top_onion=True)
            clean_task = f"بناءً على بيانات الشبكة المظلمة التالية حول ({query})، لخص النتائج وقدم التحليل المطلوب:"

        prompt = (
            f"<|bos|><|sys|>أنت CyberCore-Zero 1.5B خبير الأمن السيبراني والبايثون لبيئة {env}. "
            f"يمكنك استخدام <|web_search|> للبحث في الإنترنت العادي أو <|tor_search|> للبحث عبر Tor عند الحاجة.<|eos|>\n"
            f"<|usr|><|env_{env}|> {clean_task}<|eos|>\n"
        )
        if injected_obs:
            prompt += f"<|obs|>\n{injected_obs[:2800]}\n<|eos|>\n"
        prompt += "<|ast|>"

        print("\n[CyberCore-Zero 1.5B]> ", end="", flush=True)
        runner.generate_with_live_search(prompt, max_new_tokens=768, temperature=0.2, stream=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=str, default="export/cybercore-1.5b-q8_0.gguf")
    parser.add_argument("--tok", type=str, default=None)
    parser.add_argument("--no_int8", action="store_true")
    args = parser.parse_args()
    interactive_chat(args.gguf, args.tok, args.no_int8)
