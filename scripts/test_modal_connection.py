"""Test Modal connection and GPU availability."""

import modal

app = modal.App("test-modal-connection")

image = modal.Image.debian_slim().pip_install("torch")

@app.function(image=image, gpu="T4")
def check_gpu():
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
    }

if __name__ == "__main__":
    with app.run():
        res = check_gpu.remote()
        print("Modal GPU Check Result:", res)
