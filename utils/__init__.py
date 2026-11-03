"""
Helpers shared by the training scripts of every dataset: the control modules that replace SPARKS'
Hebbian attention block, the --tau_s argparse action, seeding and saving the results.
"""
import argparse
import random
from typing import Optional

import numpy as np
import torch

from sparks.models import SPARKS

from controls import (CorrelationAttentionBlock, GRUControlBlock, MLPControlBlock, SymmetricHebbianAttentionBlock,
                      TransformerControlBlock)

# The modules that replace the Hebbian attention block, by --block_type (see controls)
CONTROL_BLOCKS = {'symmetric': SymmetricHebbianAttentionBlock,
                  'correlation': CorrelationAttentionBlock,
                  'mlp': MLPControlBlock,
                  'gru': GRUControlBlock,
                  'transformer': TransformerControlBlock}


class FloatOrList(argparse.Action):
    """
    argparse action for an option that takes one float or several (to use with nargs='+' and type=float),
    such as --tau_s, which gives one head per value: stores a single value as a float, and several as a
    list. Several values are separated by spaces (--tau_s 0.05 0.1 0.5); --tau_s=0.5 sets one.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values[0] if len(values) == 1 else values)


def set_seed(seed: Optional[int]) -> None:
    """
    Seeds Python's, numpy's and torch's (CPU and every GPU) random number generators, for a reproducible
    run. Does nothing when seed is None.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def save_results(results_path: str,
                 test_acc: float,
                 best_test_acc: float,
                 encoder_outputs: Optional[torch.Tensor],
                 decoder_outputs: torch.Tensor,
                 model: SPARKS) -> float:
    """
    Saves a test evaluation's outputs under results_path. When test_acc is at least the best seen
    so far, also saves test_acc and the model's weights, and the outputs are saved as the best
    ones (test_*_best.npy); otherwise they are saved as the last ones (test_*_last.npy).

    Args:
        results_path (str): The directory to save the results in.
        test_acc (float): The test accuracy of this evaluation.
        best_test_acc (float): The best test accuracy seen so far.
        encoder_outputs (torch.Tensor, optional): The encoder's outputs on the test set, or None
            to save none.
        decoder_outputs (torch.Tensor): The decoder's outputs on the test set.
        model (SPARKS): The model.

    Returns:
        float: The updated best test accuracy.
    """
    if test_acc >= best_test_acc:
        best_test_acc = test_acc
        np.save(results_path + '/test_acc.npy', test_acc)
        if encoder_outputs is not None:
            np.save(results_path + '/test_enc_outputs_best.npy', encoder_outputs.cpu().numpy())
        torch.save(model.state_dict(), results_path + '/sparks.pt')
        np.save(results_path + '/test_dec_outputs_best.npy', decoder_outputs.cpu().numpy())
    else:
        if encoder_outputs is not None:
            np.save(results_path + '/test_enc_outputs_last.npy', encoder_outputs.cpu().numpy())
        np.save(results_path + '/test_dec_outputs_last.npy', decoder_outputs.cpu().numpy())

    return best_test_acc
