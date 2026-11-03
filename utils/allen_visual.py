"""
Helpers for the Allen Brain Observatory natural movie one experiments: building the control network, its
optimizer, scheduler and loss, the per-mode targets and the 1-s frame decoding accuracy. Used by
scripts/allen_visual/train_allen_movies_pseudomouse.py.
"""
from typing import Union

import numpy as np
import torch

from sparks.models import SPARKS, HebbianAttentionConfig, LinearDecoder, MLPDecoder, SharedConfig
from sparks.training import TemporalCrossEntropyLoss, build_param_groups, get_lr_multiplier

from utils import CONTROL_BLOCKS


def make_network_and_optimizers(args, datasets, frames):
    """
    Builds the network (SPARKS with its Hebbian attention block replaced by the module of args.block_type),
    its AdamW optimizer, learning-rate scheduler and loss.

    Args:
        args (argparse.Namespace): The script's arguments (architecture, block_type, mode, lr, n_epochs,
            batch_size, label_smoothing, kl_type and device).
        datasets (List[torch.utils.data.Dataset]): One training dataset per session; their units
            set the number of input neurons and their lengths the number of scheduler steps.
        frames (torch.Tensor): [T, n_pixels] movie frames; only their size is used, in
            reconstruction mode.

    Returns:
        model (SPARKS): The network, on args.device.
        optimizer (torch.optim.AdamW): The optimizer.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Linear warmup over 5% of the steps, then
            cosine decay, stepped after every batch.
        loss_fn (torch.nn.Module): Cross-entropy over the movie frames in prediction mode,
            binary cross-entropy on logits otherwise.
        kl_transition (GaussianTransition or None): The transition model of the sequential KL prior,
            None for kl_type == 'standard'.
    """
    input_sizes = [len(train_dataset.good_units_ids) for train_dataset in datasets]
    if args.mode == 'prediction':
        output_sizes = 900 * args.tau_f  # one class per frame of natural movie one (30 s at 30 fps)
    elif args.mode == 'reconstruction':
        output_sizes = frames.shape[-1] * args.tau_f
    elif args.mode in ['unsupervised', 'denoising']:
        output_sizes = [input_size * args.tau_f for input_size in input_sizes]
    else:
        raise NotImplementedError

    hebbian_config = HebbianAttentionConfig(block_class=CONTROL_BLOCKS[args.block_type], tau_s=args.tau_s,
                                            dt=args.dt, dropout=args.dropout)
    shared_config = SharedConfig(n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)

    if args.mode == 'prediction':
        decoder = LinearDecoder(in_dim=args.latent_dim * args.tau_p,
                                output_dim_per_session=output_sizes,
                                joint_decoder=True).to(args.device)
    else:
        # One output head shared by all sessions for the frames, one head per session for the spikes
        decoder = MLPDecoder(in_dim=args.latent_dim * args.tau_p,
                             hidden_dims=[args.latent_dim * args.tau_p * 2],
                             output_dim_per_session=output_sizes,
                             joint_decoder=args.mode == 'reconstruction').to(args.device)
    model = SPARKS(n_neurons_per_session=input_sizes,
                   embed_dim=args.embed_dim,
                   latent_dim=args.latent_dim,
                   tau_p=args.tau_p,
                   tau_f=args.tau_f,
                   bottleneck_dim=args.bottleneck_dim,
                   hebbian_config=hebbian_config,
                   shared_config=shared_config,
                   output_dim_per_session=output_sizes,
                   decoder=decoder,
                   device=args.device)

    params_groups, kl_transition = build_param_groups(model, kl_type=args.kl_type, weight_decay=0.01)
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr)
    steps_per_epoch = sum([np.ceil(len(dataset) / args.batch_size) for dataset in datasets])
    warmup_steps = int(0.05 * args.n_epochs * steps_per_epoch)  # 5% of total steps
    total_steps = int(args.n_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: get_lr_multiplier(step, warmup_steps,
                                                                                           total_steps=total_steps,
                                                                                           min_lr=0.0001))

    if args.mode == 'prediction':
        loss_fn = TemporalCrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    return model, optimizer, scheduler, loss_fn, kl_transition


def get_targets(targets, mode, frames=None, dt=0.01):
    """
    The targets of a batch for the given task.

    Args:
        targets (torch.Tensor): The batch's targets from the dataloader, [B, ...].
        mode (str): 'prediction': the index of the movie frame shown at each timestep, [B, T, 1];
            'reconstruction': `frames`, repeated over the batch, [B, T, n_pixels];
            'unsupervised' or 'denoising': `targets`, unchanged.
        frames (torch.Tensor, optional): [T, n_pixels] movie frames, required in reconstruction mode.
        dt (float): The time step, in seconds. Default is 0.01.

    Returns:
        torch.Tensor: The targets.
    """
    if mode == 'prediction':
        # The movie lasts 30 s at 30 fps: map each dt-long timestep to the frame shown at its end
        time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))
        frames_bin_edges = np.concatenate((np.arange(1/30, 30, 1/30), np.array([30.])))
        frames_indices_per_timestep = torch.tensor([np.where(time_bin_edges[i] <= frames_bin_edges)[0][0]
                                                    for i in range(1, len(time_bin_edges))]).type(torch.long)
        return frames_indices_per_timestep.unsqueeze(0).repeat(targets.shape[0], 1).unsqueeze(-1)
    elif mode == 'reconstruction':
        return frames.unsqueeze(0).expand(targets.shape[0], -1, -1)
    elif mode in ['unsupervised', 'denoising']:
        return targets


@torch.no_grad()
def calculate_test_acc(decoder_outputs: Union[np.ndarray, torch.Tensor], dt: float = 0.01, tol: int = 30):
    """
    The 1-s test accuracy of movie frame decoding: the fraction of timesteps at which the
    predicted frame is within `tol` frames of the frame being shown.

    Args:
        decoder_outputs (np.ndarray or torch.Tensor): [n_trials, T, 900] predictions (one score per
            movie frame).
        dt (float, optional): The time step, in seconds. Default is 0.01.
        tol (int, optional): Tolerance in number of frames. Default is 30 (1 s at 30 fps).

    Returns:
        float: The test accuracy.
    """
    if isinstance(decoder_outputs, torch.Tensor):
        decoder_outputs = decoder_outputs.cpu().numpy()

    time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))
    frames_bin_edges = np.concatenate((np.arange(1/30, 30, 1/30), np.array([30.])))

    frames_indices_per_timestep = np.array([np.where(time_bin_edges[i] <= frames_bin_edges)[0][0]
                                            for i in range(1, len(time_bin_edges))])

    target_windows = [np.arange(t - tol, t + tol)[None, :] for t in frames_indices_per_timestep]
    test_acc = np.mean(np.array([[np.isin(decoder_outputs[k, t].argmax(axis=-1), target_windows[t])
                                  for t in range(decoder_outputs.shape[1])] for k in range(decoder_outputs.shape[0])]))

    return test_acc
