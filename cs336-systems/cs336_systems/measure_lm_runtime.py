# Add the cs336-basics directory to the sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cs336-basics/')))
# print(sys.path)

import argparse
import logging
import torch
import numpy as np
import numpy.typing as npt
import timeit


from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import cs336_basics.nn_utils as nn_utils
from cs336_basics.data import get_batch
from cs336_systems.common import get_device


logger = logging.getLogger(__name__)
def configure_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")

DATASET_LEN = 10000
VOCAB_SIZE = 10000
CONTEXT_LEN = 128
BATCH_SIZE = 16
DFF_MULTIPLIER = 4

def create_model(args) -> BasicsTransformerLM:
    d_ff = 4 * args.d_model
    return BasicsTransformerLM(VOCAB_SIZE, CONTEXT_LEN, args.d_model, args.num_layers, args.num_heads, d_ff).to(get_device())

def initialize_optimizer(model: BasicsTransformerLM) -> AdamW:
    optimizer = AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The training loop for Transformer Model using AdamW optimizer."
    )

    # Model hyper-params
    parser.add_argument(
        "--d_model", type=int, required=True, help="Embedding dimensionality"
    )
    parser.add_argument(
        "--num_layers", type=int, required=True, help="Number of transformer layers."
    )
    parser.add_argument(
        "--num_heads", type=int, required=True, help="Number of heads for multi-head transformer"
    )

    # Add an argument with choices
    parser.add_argument('--model_mode', choices=['forward', 'full'], required= True,
                        help='Choose one of the three options: forward, backward, or full')

    parser.add_argument('--measure', choices=['profile', 'benchmark'], required= True,
                        help='Choose one of the three options: profile, benchmark')
   
    # Optional arguments
    parser.add_argument(
        "--debug", action="store_true", help="Include debug logging statements."
    )
    
    return parser

def forward_pass(model: BasicsTransformerLM, x: torch.LongTensor, y: torch.LongTensor) -> torch.FloatTensor:
    # Forward (compute loss)
    pred_logits = model(x)
    loss = nn_utils.cross_entropy(pred_logits, y)
    return loss

def full_pass(model: BasicsTransformerLM, x: torch.LongTensor, y: torch.LongTensor):
    loss = forward_pass(model, x, y)
    # Backward (compute gradients)
    loss.backward()
    return
    # # Clip gradients (part of optimizer)
    # nn_utils.gradient_clipping(model.parameters(), 1.0)

def benchmark(model: BasicsTransformerLM, x: torch.LongTensor, y: torch.LongTensor, run_backward = False):
    # Warmup with only forward pass
    logger.debug("Starting warmup")
    steps_warmup = 2
    for i in range(steps_warmup):
        full_pass(model, x, y)
    torch.cuda.synchronize()
    
    # Benchmark
    logger.debug("Finished warmup. Starting benchmark")
    steps_benchmark = 5
    runtimes = np.zeros(steps_benchmark)
    for i in range(steps_benchmark):
        start_t = timeit.default_timer()
        if run_backward:
            full_pass(model, x, y)
        else:
            forward_pass(model, x, y)

        torch.cuda.synchronize()
        end_t = timeit.default_timer()
        runtime = end_t - start_t
        logger.debug(f"#{i}: {runtime: .2e}s")
        runtimes[i] = runtime

    results = {
        "mean" : np.mean(runtimes),
        "std" : np.std(runtimes),
    }
    logger.info(f"Mean: {results['mean']: .2e}, Std: {results['std']: .2e}")
    return results


### Profiling code: Don't profile while benchmarking
from torch.profiler import profile, record_function, ProfilerActivity
def run_step(model, inputs, targets, optimizer, run_backward = False):
    # Aggregate activity
    with record_function('forward_pass'):
        loss = forward_pass(model, inputs, targets)
    
    if run_backward:
        with record_function('backward_pass'):
            loss.backward()
        with record_function('gradient_clipping'):
            nn_utils.clip_gradient(model.parameters(), 1.0)
        with record_function('optimizer'):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

def profile_lm(model, inputs, targets,  nsteps = 10, run_backward = False):
    optimizer = initialize_optimizer(model)
    # Warmup with one full pass
    full_pass(model, inputs, inputs)
    torch.cuda.synchronize()

    # Profile code:
    with profile(
        activities= [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True), # ??? Not sure why we need this ???
        record_shapes=True,
        profile_memory=False,
        with_stack=True # So we can capture and export to flame-graph
    ) as prof:
        for _ in range(nsteps):
            run_step(model, inputs, targets, optimizer, run_backward)
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
            # Marks the beginning and potentially the end of the code region you want to profile
            prof.step()
    
    # Export CUDA stacks (ignores CPU ones)
    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")

    # Print output
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    configure_logging(args.debug)
        
    # Create model and benchmark
    transformer_lm = create_model(args)
    logger.debug("Loading dataset warmup")
    dataset = np.random.randint(0, VOCAB_SIZE, size = DATASET_LEN)

    # Generate random data. Note that the CPU -> GPU data loading is async
    input_batch, target_batch = get_batch(dataset, CONTEXT_LEN, BATCH_SIZE, str(get_device()))
    logger.debug(f"Benchmarking model in {args.model_mode} mode.")

    # Run benchmark or profiler
    if args.measure == 'benchmark':
        benchmark(transformer_lm, input_batch, target_batch, run_backward=args.model_mode=='full')
    else:
        profile_lm(transformer_lm, input_batch, target_batch, nsteps=15, run_backward=args.model_mode=='full')

