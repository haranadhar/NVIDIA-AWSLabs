outputs/aws-nvidia-job-labs/gpu_smoke_test.py

"""GPU readiness check: CUDA detection, tensor work, and VRAM evidence."""
import json
import time

import torch


if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: check AMI driver, container runtime, and PyTorch CUDA build.")

device = torch.device("cuda:0")
torch.cuda.reset_peak_memory_stats(device)
a = torch.randn((4096, 4096), device=device, dtype=torch.float16)
b = torch.randn((4096, 4096), device=device, dtype=torch.float16)
torch.cuda.synchronize(device)
started = time.perf_counter()
c = a @ b
torch.cuda.synchronize(device)
elapsed = time.perf_counter() - started

print(json.dumps({
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "matrix_size": 4096,
    "elapsed_ms": round(elapsed * 1000, 2),
    "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
    "result_checksum": round(float(c.float().mean()), 5),
}, indent=2))
