"""
Trains a control of SPARKS on the Neural Latents Benchmark area2_bump dataset (DANDI 000127): SPARKS with its
Hebbian attention block replaced by another module (--block_type, see the controls package). The data are the
spikes of 65 neurons of somatosensory area 2 of monkey Han during centre-out reaches, in 1 ms bins, one trial per
active reach from 100 ms before to 500 ms after movement onset. --mode sets the task: in 'prediction', the model
predicts a behavioural variable (--target_type) 40 ms after the spikes; in 'co-smoothing' (the NLB task), it
predicts every neuron's spikes from the held-in neurons' ones; in 'unsupervised', it predicts its input spikes.
The model is tested on the NLB validation trials (on every trial in 'unsupervised' mode), and scored with
utils.monkey.get_accuracy.

Run from the repository root (see paper_configurations.sh for the runs of the paper), e.g.:

    python -m scripts.monkey.area2_bump --home=/path/to/data --n_epochs=1000 \
        --batch_size=64 --lr=0.01 --num_workers=4 --block_type=gru --seed=1

Saves, under results/monkey/<block_type>/<mode>/ in the current directory, a
<NNN>__<date>_area2_bump_<mode>_e_<embed_dim>_b_<bottleneck_dim>_d_<latent_dim>/ folder with the arguments
(commandline_args.txt), the number of trainable weights (model_config.npy) and, at every test, the latents and
decoder outputs on the test set (test_*_best.npy or test_*_last.npy), the best test score (test_acc.npy) and the
best model's weights (sparks.pt).
"""
import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.nlb.area2_bump import make_area2_bump_data
from sparks.models import SPARKS, HebbianAttentionConfig, LinearDecoder, SharedConfig
from sparks.training import TemporalCrossEntropyLoss, build_param_groups, evaluate, get_lr_multiplier, train_on_batch
from sparks.training.experiment import EarlyStopper, make_res_folder

from utils import CONTROL_BLOCKS, FloatOrList, save_results, set_seed
from utils.monkey import get_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a control of SPARKS on the NLB Area2_Bump dataset '
                                                 '(monkey reaching)')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=1, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.0001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=141, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience')
    parser.add_argument('--eps', type=float, default=0.001, help='early stopping tolerance')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=2, help='Size of the latent space')
    parser.add_argument('--bottleneck_dim', type=int, default=8, help='Size of the bottleneck layer')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=32, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    parser.add_argument('--tau_p', type=int, default=10, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--w_plus', type=float, default=0.001, help='STDP update magnitude')
    parser.add_argument('--tau_s', type=float, nargs='+', action=FloatOrList, default=0.5,
                        help='STDP decay, or several (space-separated) for one head each; set to 0.01 0.05 0.1 0.5 '
                             'in the co-smoothing mode')
    parser.add_argument('--kl_type', type=str, default='sequential', choices=['sequential', 'standard'])
    parser.add_argument('--block_type', type=str, default='symmetric', choices=list(CONTROL_BLOCKS),
                        help='The module that replaces the Hebbian attention block')
    parser.add_argument('--cd_rate', type=float, default=0., help='Coordinated spike dropout rate')

    # Data parameters
    parser.add_argument('--target_type', type=str, default='hand_pos', choices=['hand_pos', 'hand_vel',
                                                                                'force', 'muscle_len',
                                                                                'direction', 'joint_ang'])
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prediction', 'unsupervised', 'co-smoothing'], help='Which type of task to perform')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed of every random number generator; default None (unseeded)')

    args = parser.parse_args()
    set_seed(args.seed)

    # In co-smoothing, the block has one head per time constant (before make_res_folder, so that
    # commandline_args.txt records them)
    if args.mode == 'co-smoothing':
        args.tau_s = [0.01, 0.05, 0.1, 0.5]

    # Create folder to save results
    make_res_folder('area2_bump' + f'_{args.mode}_e_{args.embed_dim}_b_{args.bottleneck_dim}_d_{args.latent_dim}',
                    os.path.join(os.getcwd(), f'results/monkey/{args.block_type}/{args.mode}/'), args)

    # Create dataloaders
    data_path = os.path.join(args.home, "datasets/000127/sub-Han/sub-Han_desc-train_behavior+ecephys.nwb")
    (train_dataset, test_dataset,
     train_dl, test_dl) = make_area2_bump_data(data_path,
                                               mode=args.mode,
                                               y_keys=args.target_type,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

    # Make network
    if args.mode in ['prediction', 'co-smoothing']:
        output_size = train_dataset.target_data.shape[-1]
    elif args.mode == 'unsupervised':
        output_size = train_dataset.n_neurons
    else:
        raise NotImplementedError

    hebbian_config = HebbianAttentionConfig(block_class=CONTROL_BLOCKS[args.block_type], tau_s=args.tau_s,
                                            dt=args.dt, dropout=args.dropout, w_plus=args.w_plus)
    shared_config = SharedConfig(n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)

    # In the unsupervised and co-smoothing modes, the output heads start out predicting a spike
    # probability of 0.01 (see base_rate in sparks.models.LinearDecoder)
    decoder = LinearDecoder(in_dim=args.latent_dim * args.tau_p,
                            output_dim_per_session=output_size * args.tau_f,
                            base_rate=0.01 if args.mode in ['unsupervised', 'co-smoothing'] else 0.,
                            joint_decoder=True).to(args.device)
    model = SPARKS(n_neurons_per_session=train_dataset.n_neurons,
                   embed_dim=args.embed_dim,
                   bottleneck_dim=args.bottleneck_dim,
                   latent_dim=args.latent_dim,
                   tau_p=args.tau_p,
                   tau_f=args.tau_f,
                   hebbian_config=hebbian_config,
                   shared_config=shared_config,
                   decoder=decoder,
                   output_dim_per_session=output_size * args.tau_f,
                   device=args.device)

    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    np.save(os.path.join(args.results_path, 'model_config.npy'), np.array([num_weights], dtype=object))

    # Training parameters
    params_groups, kl_transition = build_param_groups(model, kl_type=args.kl_type, weight_decay=0.01)
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr)

    warmup_steps = int(0.05 * args.n_epochs * len(train_dl))  # 5% of total steps
    total_steps = args.n_epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: get_lr_multiplier(step, warmup_steps,
                                                                                           total_steps=total_steps,
                                                                                           min_lr=0.0001))
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.eps)

    # In co-smoothing, the loss is averaged over the entries selected by cd_mask, so it is left unreduced
    if args.mode == 'co-smoothing':
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.target_type == 'direction':
        loss_fn = TemporalCrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    # Per-entry losses, which train_on_batch averages over the entries selected by cd_mask
    masked_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    best_test_acc = -np.inf

    # Training loop
    pbar = tqdm.tqdm(range(args.n_epochs))
    for epoch in pbar:
        for inputs, targets in train_dl:
            if args.mode == 'co-smoothing':
                # Coordinated dropout: each input is zeroed with probability cd_rate (the others rescaled to
                # keep their mean). The loss covers only the zeroed inputs, so that the model cannot just copy
                # its inputs, plus every entry of the held-out neurons, which come after the held-in ones in
                # targets.
                dropout_mask = torch.bernoulli(torch.ones_like(inputs) * (1 - args.cd_rate))
                cd_mask = torch.ones_like(targets).to(args.device)
                cd_mask[..., :inputs.shape[-1]] = 1 - dropout_mask
                inputs = (inputs * dropout_mask) / (1 - args.cd_rate)
            elif args.mode == 'unsupervised':
                # The model is trained to reconstruct the spikes, before any dropout. With coordinated dropout
                # (cd_rate > 0), the inputs are dropped out as in co-smoothing and the loss covers only the
                # zeroed entries.
                targets = inputs.clone()
                cd_mask = None
                if args.cd_rate > 0:
                    dropout_mask = torch.bernoulli(torch.ones_like(inputs) * (1 - args.cd_rate))
                    cd_mask = (1 - dropout_mask).to(args.device)
                    inputs = (inputs * dropout_mask) / (1 - args.cd_rate)
            else:
                cd_mask = None
            train_on_batch(model=model,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn if cd_mask is None else masked_loss_fn,
                           optimizer=optimizer,
                           transition=kl_transition,
                           beta=args.beta,
                           cd_mask=cd_mask)
            scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            # The decoder outputs logits, mapped back by the sigmoid except for the direction classes, which are
            # scored by their argmax
            if args.mode == 'prediction' and args.target_type == 'direction':
                act = None
            else:
                act = torch.sigmoid

            _, encoder_outputs, decoder_outputs = evaluate(model=model,
                                                           test_dls=[test_dl],
                                                           unsupervised=args.mode == 'unsupervised',
                                                           act=act)
            encoder_outputs, decoder_outputs = encoder_outputs[0], decoder_outputs[0]

            test_acc = get_accuracy(decoder_outputs, test_dataset, args.mode, args.target_type)
            pbar.set_description("Epoch %d, acc: %.3f" % (epoch, test_acc))

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc,
                                         encoder_outputs, decoder_outputs, model)

            if early_stopper.early_stop(test_acc):
                print(f'Training stopped at epoch {epoch+1}')
                break
