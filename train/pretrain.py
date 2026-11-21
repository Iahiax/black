# train/pretrain.py
import os
import gc
import math
import time
import argparse
import numpy as np
import torch
from config import CyberCoreConfig
from model.architecture import CyberCoreZero

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

def auto_tune_for_colab_gpu():
    """ضبط الإعدادات تلقائياً حسب بطاقة Colab المتاحة (T4 / L4 / A100)"""
    if not torch.cuda.is_available():
        return 512, 1, 32, False
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"[*] بطاقة GPU المكتشفة: {props.name} ({vram_gb:.1f} GB VRAM) | BF16: {bf16_ok}")

    if vram_gb >= 35:      # A100 40GB / 80GB
        return 2048, 2, 16, bf16_ok
    elif vram_gb >= 20:    # L4 24GB
        return 1024, 1, 32, bf16_ok
    else:                  # T4 15GB (وضع توفير الذاكرة القصوى لنموذج 1.5B)
        return 512, 1, 32, False

def get_batch(data_memmap, batch_size, seq_len, device):
    max_start = len(data_memmap) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"ملف train.bin صغير جداً (يحتاج أكثر من {seq_len} رمز). أضف بيانات عبر harvester.py.")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(data_memmap[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_memmap[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)

def save_checkpoints(model, optimizer, cfg, step, ckpt_dir):
    latest_ckpt = os.path.join(ckpt_dir, "cybercore_1.5b_latest.pt")
    infer_ckpt = os.path.join(ckpt_dir, "cybercore_1.5b_inference.pt")

    raw_state = model.state_dict()
    torch.save({
        "step": step,
        "config": cfg.to_dict(),
        "model": raw_state,
        "optimizer": optimizer.state_dict(),
    }, latest_ckpt)

    # نسخة التشغيل الخفيفة للـ CPU (~3GB FP16 بدون تكرار lm_head المرتبطة)
    slim_state = {
        k: v.detach().to(dtype=torch.float16, device="cpu")
        for k, v in raw_state.items()
        if not (cfg.tie_word_embeddings and k == "lm_head.weight")
    }
    torch.save({"config": cfg.to_dict(), "model": slim_state}, infer_ckpt)
    del slim_state
    gc.collect()
    print(f"[+] تم الحفظ بنجاح | استكمال: {latest_ckpt} | تشغيل CPU: {infer_ckpt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.bin")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=0, help="0 = ضبط تلقائي حسب GPU Colab")
    parser.add_argument("--batch_size", type=int, default=0, help="0 = ضبط تلقائي حسب GPU Colab")
    parser.add_argument("--grad_accum", type=int, default=0, help="0 = ضبط تلقائي حسب GPU Colab")
    parser.add_argument("--save_every", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    auto_seq, auto_bs, auto_accum, use_bf16 = auto_tune_for_colab_gpu()

    seq_len = args.seq_len or auto_seq
    batch_size = args.batch_size or auto_bs
    grad_accum = args.grad_accum or auto_accum
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    cfg = CyberCoreConfig()
    model = CyberCoreZero(cfg, gradient_checkpointing=True).to(device)
    print(f"[*] {cfg.model_name} | المعاملات الفريدة: {model.count_parameters()/1e9:.3f}B")
    print(f"[*] إعدادات التدريب: seq_len={seq_len}, batch_size={batch_size}, grad_accum={grad_accum}, dtype={amp_dtype}")

    if device == "cuda" and HAS_BNB:
        # PagedAdamW8bit ينقل حالات المحسّن تلقائياً إلى CPU RAM عند امتلاء VRAM (أساسي لبطاقة T4)
        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        print("[+] تم تفعيل PagedAdamW8bit لتشغيل 1.5B بأمان على Google Colab.")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    latest_ckpt = os.path.join(args.ckpt_dir, "cybercore_1.5b_latest.pt")
    start_step = 0
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0) + 1
        del ckpt
        torch.cuda.empty_cache()
        print(f"[+] استكمال تلقائي للتدريب من الخطوة {start_step}")

    data = np.memmap(args.data_path, dtype=np.uint32, mode="r")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

    model.train()
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, warmup_steps=200, max_steps=args.max_steps, max_lr=3e-4, min_lr=3e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(grad_accum):
            x, y = get_batch(data, batch_size, seq_len, device)
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()
            loss_accum += loss.item()

        if device == "cuda":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % 20 == 0 or step == args.max_steps - 1:
            dt = time.time() - t0
            t0 = time.time()
            mem = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0.0
            print(f"Step {step:5d}/{args.max_steps} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | VRAM Peak: {mem:.1f}GB | Time: {dt:.1f}s")

        if (step > 0 and step % args.save_every == 0) or step == args.max_steps - 1:
            save_checkpoints(model, optimizer, cfg, step, args.ckpt_dir)

if __name__ == "__main__":
    main()
