# train/pretrain.py
import os
import math
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import CyberCoreConfig
from model.architecture import CyberCoreZero

def get_batch(data_memmap, batch_size, seq_len, device):
    ix = torch.randint(len(data_memmap) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data_memmap[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_memmap[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(data_path="data/train.bin", ckpt_dir="checkpoints", resume_path=None, max_steps=10000):
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        master_process = True

    os.makedirs(ckpt_dir, exist_ok=True)
    cfg = CyberCoreConfig()
    model = CyberCoreZero(cfg).to(device)

    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        if master_process:
            print(f"[+] تم استكمال التدريب من النقطة: {resume_path}")

    if master_process:
        print(f"[*] تهيئة CyberCore-Zero من الصفر | عدد المعاملات: {model.count_parameters()/1e9:.2f}B")

    # تسريع PyTorch 2.x
    if torch.cuda.is_available():
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=torch.cuda.is_available())

    data = np.memmap(data_path, dtype=np.uint32, mode="r")
    seq_len = 2048      # ارفعه تدريجياً إلى 8192 ثم 16384 في المراحل اللاحقة
    batch_size = 4
    grad_accum = 16     # Batch فعلي = batch_size * grad_accum * num_gpus

    scaler = torch.amp.GradScaler("cuda", enabled=False) # BF16 لا يحتاج Scaler
    t0 = time.time()

    for step in range(max_steps):
        lr = get_lr(step, warmup_steps=500, max_steps=max_steps, max_lr=3e-4, min_lr=3e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(grad_accum):
            x, y = get_batch(data, batch_size, seq_len, device)
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if master_process and (step % 50 == 0 or step == max_steps - 1):
            dt = time.time() - t0
            t0 = time.time()
            print(f"Step {step:5d} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Time: {dt:.2f}s")

        if master_process and step > 0 and step % 1000 == 0:
            save_file = os.path.join(ckpt_dir, f"cybercore_step_{step}.pt")
            state = raw_model._orig_mod.state_dict() if hasattr(raw_model, "_orig_mod") else raw_model.state_dict()
            torch.save(state, save_file)
            print(f"[+] حفظ نقطة التفتيش: {save_file}")

    if master_process:
        final_file = os.path.join(ckpt_dir, "cybercore_final.pt")
        state = raw_model._orig_mod.state_dict() if hasattr(raw_model, "_orig_mod") else raw_model.state_dict()
        torch.save(state, final_file)

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
