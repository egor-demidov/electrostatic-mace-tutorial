import torch
from mace.calculators import mace_polar

# Check if CUDA is available, fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

calc = mace_polar(
    model="polar-1-m",  # or "polar-1-l"
    device=device,
    default_dtype="float32",  # use float32 for faster MD, float64 for minimization
    enable_cueq=True, enable_oeq=True
)