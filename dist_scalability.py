# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model.transformer import ParallelAttention, ParallelMLP
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch

def model_provider(model_arch):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building GPT model ...')
    if model_arch == "attention":
        model = ParallelAttention(config)
    elif model_arch == "mlp":
        model = ParallelMLP(config)

    return model

def add_scale_args(parser):
    group = parser.add_argument_group(title='model scalability')

    group.add_argument("--arch", type=str, default="attention", choices=["attention", "mlp"],
                       help='Model architecture.')
    group.add_argument("--bs", type=int, default=1,
                       help='batch size')
    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_scale_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")

    model = get_model(model_provider, wrap_with_ddp=False)

    print(model)