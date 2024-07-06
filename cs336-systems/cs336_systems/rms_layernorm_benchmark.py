# Benchmarking just RmsNorm implementation v/s nn.LayerNorm
# Findings >> Empirically, RmsNorm is 3-4x slower than nn.LayerNorm

# End-to-end via measure_lm_runtime
# !python cs336-systems/cs336_systems/measure_lm_runtime.py --d_model=1600 --num_layers=48 --num_heads=25 --model_mode=forward --measure=benchmark --mixed_precision --use_pytorch_layernorm
# !python cs336-systems/cs336_systems/measure_lm_runtime.py --d_model=1600 --num_layers=48 --num_heads=25 --model_mode=forward --measure=benchmark --mixed_precision 
# TransformerLM (LayerNorm) - Mean:  7.89e-02, Std:  3.68e-02
# TransformerLM (Rms) - Mean:  8.51e-02, Std:  3.72e-02


import torch
import torch.nn as nn
import numpy as np
import timeit

class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        # gains are named "weight" to match dict key so we can use nn.Module.load_state_dict
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, activations: torch.Tensor):
        """
        Args:
        - activations: torch.FloatTensor
                Input features to run RMSNorm on. Tensor of (*, d_model), where *
                can be an arbitrary number of dimensions with arbitrary values.
        """
        # Unsqueeze reshapes to match dimensionalities so that broadcasting-multiplication happens row-wise
        rms_normalization = (
            activations.pow(2).mean(dim=-1).add(1e-5).rsqrt().unsqueeze(-1)
        )
        return activations * rms_normalization * self.weight


def benchmark_module(module: nn.Module, input: torch.FloatTensor, nsteps):
    # Warmup with forward passes
    for _ in range(5):
        module(input)
    torch.cuda.synchronize()

    # Benchmark
    runtimes = np.zeros(nsteps)
    for i in range(nsteps):
        start_t = timeit.default_timer()
        module(input)
        torch.cuda.synchronize()
        end_t = timeit.default_timer()
        runtimes[i] = end_t - start_t
    
    return {"mean": np.mean(runtimes), "std": np.std(runtimes)}

batch_size = 50000
feature_sizes = [1024, 2048, 4096, 8192]
nsteps = 1000

for feature_dim in feature_sizes:
    # Initlalize and move to GPU
    input = torch.randn(size=(batch_size, feature_dim), dtype=torch.float32).pin_memory().to(torch.device("cuda:0"), non_blocking=True)
    rms_module = RmsNorm(feature_dim).to(torch.device("cuda:0"))
    layer_norm = nn.LayerNorm(normalized_shape=feature_dim).to(torch.device("cuda:0"))
    print(f"Benchmarking RmsNorm for {input.shape}")
    print(benchmark_module(rms_module, input, nsteps))

    print(f"Benchmarking LayerNorm for {input.shape}")
    print(benchmark_module(layer_norm, input, nsteps))

