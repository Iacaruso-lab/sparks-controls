"""
Helpers of the MEC experiments: the bits-per-spike scores of scripts/mec/train_mec.py, and the
cross-correlation sorting of neurons used by the MEC notebooks.
"""
import numpy as np
from scipy.signal import convolve, correlate, correlation_lags
from scipy.stats import norm

from sparks.training import bits_per_spike


def spikes_downsample(spikes, downsampling_factor, mode='mean'):
    """
    Bins [n_neurons, T] spikes in windows of downsampling_factor steps: the mean number of spikes
    per step ('mean') or whether there was any ('max').
    """
    bins = np.arange(0, spikes.shape[-1] + downsampling_factor, downsampling_factor)
    spike_times = spikes * np.arange(spikes.shape[-1])[None, :]
    if mode == 'mean':
        return np.vstack([(np.histogram(spike_times_unit[spike_times_unit != 0], bins)[0]
                           / downsampling_factor)[None, :] for spike_times_unit in spike_times]).astype(float)
    elif mode == 'max':
        return np.vstack([(np.histogram(spike_times_unit[spike_times_unit != 0], bins)[0] > 0)[None, :]
                          for spike_times_unit in spike_times]).astype(float)


def fire_rate(calcium_activity, bin_size):
    """The activity of one neuron smoothed by a normalised Gaussian kernel of width 4 * bin_size."""
    kernel_width = 4 * bin_size
    x = np.linspace(-kernel_width, kernel_width, num=2 * kernel_width, endpoint=False)
    gaussian_kernel = norm.pdf(x, scale=kernel_width)
    gaussian_kernel /= np.sum(gaussian_kernel)

    return convolve(calcium_activity, gaussian_kernel, mode='same')


def sorting_xcorr(spikes, maxlag, downsampling_factor, dt_sf):
    """
    Sorts [n_neurons, T] spikes by the lag of their peak cross-correlation (within maxlag) with a
    reference neuron, that of the most strongly correlated pair.

    Returns:
        np.ndarray: The neurons' order.
    """
    FRp = spikes_downsample(spikes, downsampling_factor)
    FR = np.vstack([fire_rate(FRp[i], dt_sf) for i in range(len(FRp))])
    corr_mat = np.zeros([spikes.shape[0], spikes.shape[0]])

    # Signed peak cross-correlation: positive when neuron i leads neuron j
    for i in range(FR.shape[0]):
        for j in range(i + 1, FR.shape[0]):
            val = correlate(FR[i], FR[j], mode='full')
            time = correlation_lags(len(FR[i]), len(FR[j]), mode='full')

            idx = np.argmax(val[np.abs(time) < maxlag])
            v = val[np.abs(time) < maxlag][idx]
            t = time[np.abs(time) < maxlag][idx]

            if t >= 0:
                corr_mat[i, j] = v
                corr_mat[j, i] = -v
            else:
                corr_mat[i, j] = -v
                corr_mat[j, i] = v

    r, c = np.unravel_index(np.argmax(corr_mat), corr_mat.shape)
    return np.argsort(corr_mat[c])


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
