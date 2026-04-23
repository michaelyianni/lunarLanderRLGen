import zipfile
import torch
import io
import os

MODEL_PATH = "results/models/dqn_standard.zip"

print("Inspecting contents of zip file...\n")
with zipfile.ZipFile(MODEL_PATH, "r") as z:
    for name in z.namelist():
        info = z.getinfo(name)
        print(f"  {name:<35} {info.file_size / 1024:.2f} KB")

    # Try to manually load policy.pth
    print("\nAttempting to manually read policy.pth...")
    with z.open("policy.pth") as f:
        data = f.read()
        print(f"  Raw bytes read: {len(data)}")
        try:
            buffer = io.BytesIO(data)
            result = torch.load(buffer, map_location="cpu", weights_only=False)
            print(f"  torch.load result: {type(result)}")
        except Exception as e:
            print(f"  torch.load FAILED: {e}")