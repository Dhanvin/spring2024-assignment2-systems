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
    return BasicsTransformerLM(VOCAB_SIZE, CONTEXT_LEN, args.d_model, args.num_layers, args.num_heads, d_ff)

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
    parser.add_argument('--mode', choices=['forward', 'full'], required= True,
                        help='Choose one of the three options: forward, backward, or full')

   
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

def benchmark(model: BasicsTransformerLM, x: torch.LongTensor, y: torch.LongTensor, mode: str):
    # Warmup with only forward pass
    logger.info("Starting warmup")
    steps_warmup = 2
    for i in range(steps_warmup):
        full_pass(model, x, y)
        logger.debug(f"#{i}: {runtime: .2e}s")
    torch.cuda.synchronize()
    
    # Benchmark
    logger.info("Finished warmup. Starting benchmark")
    steps_benchmark = 5
    runtimes = np.zeros(steps_benchmark)
    for i in range(steps_benchmark):
        start_t = timeit.default_timer()
        if mode == 'forward':
            forward_pass(model, x, y)
        else:
            full_pass(model, x, y)
        torch.cuda.synchronize()
        end_t = timeit.default_timer()
        runtime = end_t - start_t
        logger.debug(f"#{i}: {runtime: .2e}s")
        runtimes[i] = runtime

    results = {
        "mean" : np.mean(runtimes),
        "std" : np.std(runtimes),
    }
    logger.info(f"Mean: {results["mean"]: .2e}, Std: {results["std"]: .2e}")
    return results


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    configure_logging(args.debug)
        
    # Create model and benchmark
    transformer_lm = create_model(args)
    logger.info("Loading dataset warmup")
    dataset = np.random.randint(0, VOCAB_SIZE, size = DATASET_LEN)

    # Generate random data. Note that the CPU -> GPU data loading is async
    input_batch, target_batch = get_batch(dataset, CONTEXT_LEN, BATCH_SIZE, str(get_device()))
    logger.info(f"Benchmarking model in {args.mode} mode.")

    # Run benchmark
    benchmark(transformer_lm, input_batch, target_batch, args.mode)

