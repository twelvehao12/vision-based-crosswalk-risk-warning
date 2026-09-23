import torch


def resolve_device(preferred: str = "auto") -> str:
    preferred = preferred.lower()

    if preferred != "auto":
        return preferred

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"