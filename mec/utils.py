"""
Helpers of the MEC experiments.
"""
import numpy as np
from scipy.signal import convolve, correlate, correlation_lags
from scipy.stats import norm

from sparks.training import bits_per_spike


def bps_test(decoder_outputs, test_dataset, tau_f, last_stop, fs, test_indices, heldin_indices,
             distribution='bernoulli'):
    """
    Bits per spike of the decoder's outputs on the test trial of scripts/mec/train_mec.py, which spans
    the recording from 600 s on.

    Args:
        decoder_outputs (torch.Tensor): [1, T, tau_f * neurons] decoder outputs on the test trial,
            as rates (after the sigmoid). Only the first step of each future window is scored.
        test_dataset (torch.utils.data.Dataset): The test dataset, holding that single trial.
        tau_f (int): The size of the model's future window.
        last_stop (float): The end of the training window, in seconds.
        fs (float): The sampling rate of the (downsampled) spikes, in Hz.
        test_indices (np.ndarray): The neurons scored after last_stop (the held-out neurons in
            co-smoothing).
        heldin_indices (np.ndarray): The neurons scored over the whole test trial.
        distribution (str): The likelihood of bits_per_spike. Default is 'bernoulli'.

    Returns:
        full_acc (float): Bits per spike of the heldin_indices neurons over the whole test trial.
        test_acc (float): Bits per spike of the test_indices neurons after last_stop.
    """
    B, T, _ = decoder_outputs.shape
    # The index of last_stop in the test trial, which starts 600 s into the recording
    last_stop_idx = int((last_stop - 600) * fs)
    decoder_outputs_full = decoder_outputs.cpu().reshape(B, T, tau_f, -1)[:, :, 0, heldin_indices]
    decoder_outputs_test = decoder_outputs.cpu().reshape(B, T, tau_f, -1)[:, last_stop_idx:, 0, test_indices]

    targets_full = test_dataset[0][0][:, heldin_indices].unsqueeze(0)
    targets_test = test_dataset[0][0][last_stop_idx:, test_indices].unsqueeze(0)

    full_acc = bits_per_spike(decoder_outputs_full, targets_full, distribution=distribution)
    test_acc = bits_per_spike(decoder_outputs_test, targets_test, distribution=distribution)

    return full_acc, test_acc
