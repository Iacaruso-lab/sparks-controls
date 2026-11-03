"""Test score of the NLB Area2_Bump experiments. Used by scripts/monkey/area2_bump.py."""
import numpy as np
from sklearn.metrics import r2_score

from sparks.training import bits_per_spike


def get_accuracy(decoder_outputs, test_dataset, mode, target_type):
    """
    Scores the decoder's outputs on the test set. Each trial spans 100 ms before to 500 ms after
    movement onset in 1 ms bins (600 bins), and only the bins from movement onset on are scored.
    With tau_f > 1, only the first step of each future window is scored, except for
    target_type='direction', where the argmax runs over every output.

    Args:
        decoder_outputs (torch.Tensor): [n_trials, 600, ...] decoder outputs on the test set, after
            the sigmoid (logits for target_type='direction').
        test_dataset (sparks.data.nlb.area2_bump.Area2BumpDataset): The test set.
        mode (str): 'prediction', 'co-smoothing' or 'unsupervised', see scripts/monkey/area2_bump.py.
        target_type (str): The behavioural variable predicted in 'prediction' mode.

    Returns:
        float: In 'prediction' mode, the fraction of bins whose reach direction is classified
            correctly for target_type='direction', and the variance-weighted R2 of the predictions
            otherwise; in 'co-smoothing' mode, the bits per spike (Bernoulli likelihood) of the
            held-out neurons; in 'unsupervised' mode, those of every neuron.
    """
    if mode == 'prediction':
        targets = test_dataset.target_data.numpy()
        if target_type == 'direction':
            preds = decoder_outputs[:, 100:].cpu().numpy()
            test_acc = np.mean(targets[:, 100:] == preds.argmax(-1))
        else:
            preds = decoder_outputs[:, 100:].cpu().numpy().reshape([targets.shape[0], targets.shape[1] - 100, -1,
                                                                    targets.shape[2]])
            test_acc = r2_score(targets[:, 100:].reshape(-1, targets.shape[-1]),
                                preds[:, :, 0].reshape(-1, targets.shape[-1]), multioutput='variance_weighted')
    elif mode in ['co-smoothing', 'unsupervised']:
        if mode == 'co-smoothing':
            targets = test_dataset.target_data[:, 100:, test_dataset.heldout_neurons]
            # 65 neurons in all: the held-in ones, then the held-out ones
            preds = decoder_outputs.cpu().reshape([targets.shape[0], 600, -1, 65])[:, 100:, 0,
                                                                                    test_dataset.heldout_neurons]
        else:
            targets = test_dataset.spikes[:, 100:]
            preds = decoder_outputs.cpu().reshape([targets.shape[0], 600, -1, targets.shape[-1]])[:, 100:, 0]

        test_acc = bits_per_spike(preds, targets, distribution='bernoulli')

    return test_acc
