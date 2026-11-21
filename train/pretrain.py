# train/pretrain.py
import os
import math
import time
import argparse
import numpy as np
import torch
from config import get_config
from model.architecture import CyberCoreZero

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

def get_batch(data_memmap, batch_size, seq_len, device):
    max_start = len(data_memmap) - seq_len - 1
    if max_start <= 0:
        raise ValueError("ملف train.bin أصغر من seq_len! أضف المزيد من البيانات عبر harvester.py.")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="colab_t4", choices=["colab_t4", "colab_a100", "heavy_7b"])
    parser.add_argument("--data_path", type=str, default="data/train.bin")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    cfg = get_config(args.profile)
    model = CyberCoreZero(cfg, gradient_checkpointing=True).to(device)
    print(f"[*] الوضع: {args.profile} | المعاملات: {model.count_parameters()/1e6:.1f}M | الجهاز: {device}")

    if device == "cuda" and HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        print("[+] تم تفعيل 8-bit AdamW لتقليل استهلاك VRAM على Colab.")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # الاستكمال التلقائي (Auto-Resume) في حال فصلت جلسة Google Colab
    latest_ckpt = os.path.join(args.ckpt_dir, "cybercore_latest.pt")
    start_step = 0
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0) + 1
        print(f"[+] استكمال تلقائي من الخطوة {start_step} من: {latest_ckpt}")

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

        for _ in range(args.grad_accum):
            x, y = get_batch(data, args.batch_size, args.seq_len, device)
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
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
            print(f"Step {step:5d}/{args.max_steps} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Time: {dt:.2f}s")

        if (step > 0 and step % args.save_every == 0) or step == args.max_steps - 1:
            payload = {
                "step": step,
                "config": cfg.to_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(payload, latest_ckpt)
            # حفظ نسخة خفيفة للتشغيل المباشر على CPU بدون حالة المحسّن (Optimizer)
            infer_path = os.path.join(args.ckpt_dir, "cybercore_inference.pt")
            torch.save({"config": cfg.to_dict(), "model": {k: v.half().cpu() for k, v in model.state_dict().items()}}, infer_path)
            print(f"[+] تم الحفظ في {latest_ckpt} + نسخة التشغيل الخفيفة في {infer_path}")

if __name__ == "__main__":
    main()
