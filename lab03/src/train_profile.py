import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.feedforward import TughanDensePerceptron
from models.convnet import TughanConvNet
from models.transformer import TughanTransformerClassifier
from models.profiling import profile_model_efficiency

def run():
    print("[INFO] Auditing Tughan Kurtulus Models...")
    models = {
        "MLP (Dense)": TughanDensePerceptron(),
        "CNN (Hierarchical)": TughanConvNet(),
        "ViT (Patch-16)": TughanTransformerClassifier()
    }
    
    metrics = {}
    for name, m in models.items():
        res = profile_model_efficiency(m)
        metrics[name] = res
        print(f"[{name}] Params: {res['parameters']:,} | MACs: {res['macs']:,} | Latency: {res['latency_ms']:.3f} ms")
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmarks.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SUCCESS] Exported results to {out_dir}")

if __name__ == "__main__":
    run()
