#!/usr/bin/env python3
import os

# CUDA library paths for GB10 (aarch64 + CUDA 13.0)
_cuda_libs = [
    "/usr/lib/aarch64-linux-gnu/libcusparseLt/13",
    "/usr/lib/aarch64-linux-gnu/nvshmem/13",
    "/usr/local/cuda-13.0/targets/sbsa-linux/lib",
    "/usr/lib/aarch64-linux-gnu",
]
os.environ["LD_LIBRARY_PATH"] = ":".join(_cuda_libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "webapp.main:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )
