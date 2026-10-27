import time
import torch
from thop import profile as thop_profile

def profile_model_efficiency(model, input_shape=(1, 1, 28, 28), iters=70):
    model.eval()
    x = torch.randn(*input_shape)
    
    n_params = sum(p.numel() for p in model.parameters())
    macs, _ = thop_profile(model, inputs=(x,), verbose=False)
    
    with torch.no_grad():
        for _ in range(15):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        elapsed = time.perf_counter() - t0
        
    latency = (elapsed / iters) * 1000.0
    return {
        "parameters": n_params,
        "macs": int(macs),
        "mflops": (macs * 2) / 1e6,
        "latency_ms": latency
    }
