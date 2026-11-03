"""
Trains a control of SPARKS on calcium imaging of mouse medial entorhinal cortex (MEC): SPARKS with its Hebbian
attention block replaced by another module (--block_type, see the controls package). The data are the binarised
deconvolved spikes of one session (animal 60584, session 17), sampled at 7.73 Hz and downsampled by --ds, used as
a single trial from 600 s to 3600 s. The model is trained to predict its input spikes, and --mode sets what is
held out: in 'co-smoothing', it trains up to --last_stop with a random --p_left_out fraction of the neurons hidden
from its inputs, and it is scored on predicting those neurons after --last_stop; in 'unsupervised', it trains and
is scored on the whole recording. Scores are bits per spike (see mec.utils.bps_test).

Run from the repository root (see paper_configurations.sh for the runs of the paper), e.g.:

    python -m scripts.mec.train_mec --home=/nemo/lab/iacarusof/home/shared/Nicolas --tau_p=1 --tau_f=1 \
        --beta=0.0001 --lr=0.005 --mode=co-smoothing --n_epochs=5000 --block_type=gru --seed=1

Saves, under results/mec/<mode>/<block_type>/tau_p/<tau_p>/tau_f/<tau_f>/ in the current directory, a
<NNN>__<date>_mec__beta_<beta>_taus_<tau_s>_tauf_<tau_f>_taup_<tau_p>/ folder with the arguments
(commandline_args.txt), the left-out neurons (left_out_indices.npy, co-smoothing only) and, at every test, the
latents and decoder outputs on the test trial (test_*_best.npy or test_*_last.npy), the best test score
(test_acc.npy) and the best model's weights (sparks.pt).
"""
import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.mec import make_mec_data
from sparks.models import SPARKS, HebbianAttentionConfig, LinearDecoder, SharedConfig
from sparks.training import build_param_groups, evaluate_on_batch, get_lr_multiplier, train_on_batch
from sparks.training.experiment import EarlyStopper, make_res_folder

from mec.utils import bps_test
from utils import CONTROL_BLOCKS, FloatOrList, save_results, set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a control of SPARKS on the MEC calcium imaging recording')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.0001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')

    parser.add_argument('--latent_dim', type=int, default=2, help='Size of the latent space')
    parser.add_argument('--bottleneck_dim', type=int, default=8, help='Size of the bottleneck layer')
    parser.add_argument('--embed_dim', type=int, default=32, help='Size of attention embeddings')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--tau_p', type=int, default=1, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, nargs='+', action=FloatOrList, default=500.,
                        help='STDP decay, or several (space-separated) for one head each; set to 10 50 100 500 '
                             'in the co-smoothing mode')
    parser.add_argument('--cd_rate', type=float, default=0.4, help='Coordinated spike dropout rate for co-smoothing')
    parser.add_argument('--config', type=str, default='light', choices=['light', 'full'])
    parser.add_argument('--last_stop', type=int, default=2600, help='Last stop time for training')
    parser.add_argument('--p_left_out', type=float, default=0.1,
                        help='Proportion of neurons to leave out for co-smoothing')
    parser.add_argument('--kl_type', type=str, default='sequential', choices=['sequential', 'standard'],
                        help='Type of KL divergence to use')
    parser.add_argument('--chunk_length', type=int, default=1000, help='Length of chunks')
    parser.add_argument('--block_type', type=str, default='symmetric', choices=list(CONTROL_BLOCKS),
                        help='The module that replaces the Hebbian attention block')

    # Data parameters
    parser.add_argument('--ds', type=int, default=4, help='downsampling factor')
    parser.add_argument('--mode', type=str, default='co-smoothing',
                        choices=['unsupervised', 'co-smoothing'], help='Which type of task to perform')
    parser.add_argument('--patience', type=int, default=500, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed of every random number generator; default None (unseeded)')

    args = parser.parse_args()
    set_seed(args.seed)

    # The recording is sampled at 7.73 Hz, downsampled by args.ds
    args.dt = args.ds / 7.73
    fs = 7.73 / args.ds

    # In co-smoothing, the block has one head per time constant (before make_res_folder, so that
    # commandline_args.txt records them)
    if args.mode == 'co-smoothing':
        args.tau_s = [10, 50, 100, 500]

    # Create folder to save results
    make_res_folder(f'mec__beta_{args.beta}_taus_{args.tau_s}_tauf_{args.tau_f}_taup_{args.tau_p}',
                    os.path.join(os.getcwd(), f'results/mec/{args.mode}/{args.block_type}/tau_p/{args.tau_p}/'
                                              f'tau_f/{args.tau_f}'), args)

    # Train and test trials are [start, stop] times in seconds. The test trial is the whole recording
    # used (600 s to 3600 s); scores are computed after args.last_stop (see mec.utils.bps_test)
    data_path = os.path.join(args.home, 'datasets/mec/calcium_activity_matrix_60584_session17.mat')
    if args.mode == 'co-smoothing':
        start_stop_times_train = np.array([(600, args.last_stop)])
        start_stop_times_test = np.array([[600, 3600]])
        p_left_out = args.p_left_out
    elif args.mode == 'unsupervised':
        start_stop_times_train = np.array([(600, 3600)])
        start_stop_times_test = np.array([[600, 3600]])
        p_left_out = 0.
        args.last_stop = 600  # score the whole test trial
    else:
        raise NotImplementedError

    train_dataset, test_dataset, train_dl, test_dl = make_mec_data(data_path,
                                                                   train_times=start_stop_times_train,
                                                                   test_times=start_stop_times_test,
                                                                   downsampling_factor=args.ds,
                                                                   p_left_out=p_left_out,
                                                                   seed=args.seed,
                                                                   batch_size=args.batch_size,
                                                                   num_workers=args.num_workers)
    left_out_indices = train_dataset.left_out_neurons  # None when no neuron is held out

    if left_out_indices is None:
        left_out_indices = np.arange(test_dataset.spikes.shape[-1])
        heldin_indices = np.arange(train_dataset.spikes.shape[-1])
    else:
        heldin_indices = np.setdiff1d(np.arange(test_dataset.spikes.shape[-1]), left_out_indices)
        np.save(os.path.join(args.results_path, 'left_out_indices.npy'), left_out_indices)

    hebbian_config = HebbianAttentionConfig(block_class=CONTROL_BLOCKS[args.block_type], dt=args.dt,
                                            tau_s=args.tau_s)
    shared_config = SharedConfig(n_layers=args.n_layers, n_heads=args.n_heads)

    input_size = len(heldin_indices)
    output_size = train_dataset.spikes.shape[-1]

    decoder = LinearDecoder(in_dim=args.latent_dim * args.tau_p,
                            output_dim_per_session=output_size * args.tau_f,
                            joint_decoder=True).to(args.device)
    model = SPARKS(n_neurons_per_session=input_size,
                   embed_dim=args.embed_dim,
                   latent_dim=args.latent_dim,
                   tau_p=args.tau_p,
                   tau_f=args.tau_f,
                   bottleneck_dim=args.bottleneck_dim,
                   hebbian_config=hebbian_config,
                   shared_config=shared_config,
                   output_dim_per_session=output_size,
                   decoder=decoder,
                   device=args.device)

    params_groups, kl_transition = build_param_groups(model, kl_type=args.kl_type, weight_decay=0.1)
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr)
    warmup_steps = int(0.05 * args.n_epochs * len(train_dl))  # 5% of total steps
    total_steps = args.n_epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: get_lr_multiplier(step, warmup_steps,
                                                                                           total_steps=total_steps,
                                                                                           min_lr=0.0001))

    # In co-smoothing, the loss is averaged over the entries selected by cd_mask, so it is left unreduced
    if args.mode == 'unsupervised':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    early_stopper = EarlyStopper(patience=args.patience)

    best_test_acc = -np.inf
    full_acc_best = -np.inf

    pbar = tqdm.tqdm(range(args.n_epochs))
    for epoch in pbar:
        for inputs, _ in train_dl:
            # The targets are every neuron's spikes; the model only sees the held-in neurons'
            targets = inputs
            if args.mode == 'co-smoothing':
                inputs = inputs[:, :, heldin_indices]
                # Coordinated dropout: each input is zeroed with probability cd_rate (the others rescaled to
                # keep their mean). The loss covers only the zeroed inputs, so that the model cannot just copy
                # its inputs, plus every entry of the held-out neurons. targets keep the recording's neuron
                # order, so the held-in neurons' entries of cd_mask are at heldin_indices.
                dropout_mask = torch.bernoulli(torch.ones_like(inputs) * (1 - args.cd_rate))
                cd_mask = torch.ones_like(targets)
                cd_mask[..., heldin_indices] = 1 - dropout_mask
                cd_mask = cd_mask.to(args.device)
                inputs = (inputs * dropout_mask) / (1 - args.cd_rate)
            else:
                inputs = inputs[:, :, heldin_indices]
                cd_mask = None
            # chunk_length: truncated backpropagation through time over the long recording, one optimizer
            # step per chunk (see sparks.training.train_on_batch)
            train_on_batch(model=model,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           transition=kl_transition,
                           beta=args.beta,
                           cd_mask=cd_mask,
                           chunk_length=args.chunk_length)
            scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            act = torch.sigmoid  # the decoder outputs logits; bits per spike scores probabilities
            decoder_outputs = torch.Tensor()

            for inputs, _ in test_dl:
                targets = inputs
                inputs = inputs[:, :, heldin_indices]
                _, encoder_outputs, decoder_outputs_batch = evaluate_on_batch(model=model,
                                                                              inputs=inputs,
                                                                              targets=targets,
                                                                              loss_fn=loss_fn,
                                                                              act=act)

                decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch.detach().cpu()), dim=0)

            # test_acc scores the left-out neurons (every neuron outside co-smoothing) after last_stop,
            # full_acc the held-in neurons over the whole test trial
            full_acc, test_acc = bps_test(decoder_outputs, test_dataset, args.tau_f,
                                          args.last_stop, fs,
                                          test_indices=left_out_indices, heldin_indices=heldin_indices)

            pbar.set_description("Epoch %d, test acc: %.3f, best test acc: %.3f, full acc: %.3f, best full acc: %.3f"
                                 % (epoch, test_acc, best_test_acc, full_acc, full_acc_best))

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc,
                                         encoder_outputs, decoder_outputs, model)

            if early_stopper.early_stop(test_acc):
                print(f'Training stopped at epoch {epoch+1}')
                break
