import torch

def get_peak_vram_mb() -> float:
    """Returns the peak GPU memory allocated in Megabytes (MB)."""
    if not torch.cuda.is_available():
        return 0.0
    # Convert bytes to MB
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def reset_memory_stats():
    """Resets the peak memory tracking. Call this right before training starts."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()