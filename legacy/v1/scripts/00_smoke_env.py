"""Day-1 smoke: verify CUDA, bitsandbytes NF4 load, and free VRAM.

Run this AFTER `pip install -r requirements.txt`. It exits non-zero if any
of the prereqs for Phase 0 are missing.
"""
import sys

def main() -> int:
    try:
        import torch
        import bitsandbytes as bnb
    except ImportError as e:
        print(f"FAIL: missing dependency {e.name}. Run: pip install -r requirements.txt")
        return 1

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available. Check driver and torch CUDA build.")
        return 1

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    total_mb = torch.cuda.get_device_properties(dev).total_memory // (1024 * 1024)
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    free_mb = free_bytes // (1024 * 1024)
    print(f"Device: {name}")
    print(f"Total VRAM: {total_mb} MB, Free: {free_mb} MB")

    if total_mb < 7500:
        print(f"FAIL: expected >= 7500 MB total VRAM, got {total_mb}")
        return 1

    # NF4 sanity: quantize a 4096x4096 tensor
    try:
        w = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
        qw = bnb.nn.Params4bit(w, quant_type="nf4").cuda(dev)
        print(f"NF4 quantize 4096x4096 OK. dtype={qw.dtype}")
    except Exception as e:
        print(f"FAIL: bitsandbytes NF4 quantize raised {type(e).__name__}: {e}")
        return 1

    cc_major, cc_minor = torch.cuda.get_device_capability(dev)
    print(f"Compute capability: {cc_major}.{cc_minor}")
    if cc_major < 8:
        print("FAIL: need Ampere or newer (compute capability >= 8.0) for fast bf16")
        return 1

    print("PASS: environment ready for Phase 0")
    return 0

if __name__ == "__main__":
    sys.exit(main())
