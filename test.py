import sys, importlib

print("Python:", sys.version.split()[0])
for pkg in ["torch", "numpy", "scipy", "matplotlib", "sklearn", "wfdb", "onnx", "onnxruntime"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:14s} {getattr(m, '__version__', 'unknown')}")
    except ImportError:
        print(f"{pkg:14s} MISSING")

try:
    import torch
    print("\nCUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
except ImportError:
    print("\ntorch not installed")