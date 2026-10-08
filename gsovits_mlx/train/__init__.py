"""Shared MLX training core for GPT-SoVITS s1/s2 trainers.

Package layout:
    mixed_precision.py -- GradScaler (torch.amp 16-mixed semantics on MLX)
    optim.py           -- AdamW (torch-exact math) and ScaledAdam (k2 port)
    schedule.py        -- WarmupCosineLRSchedule incl. the official locked-LR quirk
    loop.py            -- Trainer skeleton (fp32-master mixed precision loop)
    ckpt.py            -- resume checkpoints + s1/s2 inference-weight exporters
"""
