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


from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_systems.common import get_device


logger = logging.Logger(__name__)

DATASET_LEN = 100000
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
    parser.add_argument('--mode', choices=['forward', 'backward', 'full'],
                        help='Choose one of the three options: forward, backward, or full')

   
    # Optional arguments
    parser.add_argument(
        "--debug", action="store_true", help="Include debug logging statements."
    )
    
    return parser

def benchmark(model: BasicsTransformerLM, x: torch.LongTensor):
    # Generate random data. Note that the CPU -> GPU data loading is async

    steps_warmup = 5

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    # Set logging level based on debug argument
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")
    else:
        logger.setLevel(logging.INFO)
    
    # Create model and benchmark
    transformer_lm = create_model(args)
    dataset = np.random.randint(0, VOCAB_SIZE, size = DATASET_LEN)
    logger.info(f"Benchmarking model in {args.mode} mode.")
    device = str(get_device())
    benchmark(transformer_lm, get_batch(dataset, CONTEXT_LEN, BATCH_SIZE, device))

