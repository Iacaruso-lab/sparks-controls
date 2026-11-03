"""
Trains a control of SPARKS on a pseudo-mouse watching Allen Brain Observatory natural movie one: SPARKS with its
Hebbian attention block replaced by another module (--block_type, see the controls package), on `--n_neurons`
VISp Neuropixels units drawn at random from all sessions. Depending on --mode, the network decodes the frame
being shown (prediction, scored by the 1-s frame accuracy), reconstructs the frames (reconstruction) or predicts
the spikes themselves (unsupervised).

Run from the repository root (see paper_configurations.sh for the runs of the paper), e.g.:

    python -m scripts.allen_visual.train_allen_movies_pseudomouse --home=/nemo/lab/iacarusof/home/shared/Nicolas \
        --lr=0.01 --num_workers=4 --batch_size=5 --n_neurons=100 --block_type=gru --chunk_length=1000 \
        --mode=reconstruction --latent_dim=64

Saves the arguments, the drawn units (good_units_ids.npy), the best weights (sparks.pt) with their test accuracy
and the test encoder/decoder outputs (see utils.save_results) in
results/allen/movies/<mode>/<block_type>/block/<block>/n_neurons/<n_neurons>/
<NNN>__<date>_allen_movies_pseudomouse_<mode>_nneurons_<n_neurons>/.
"""
import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen import get_frames, make_pseudomouse_movies_data
from sparks.training import evaluate_on_batch, train_on_batch
from sparks.training.experiment import make_res_folder

from utils import CONTROL_BLOCKS, FloatOrList, save_results, set_seed
from utils.allen_visual import calculate_test_acc, get_targets, make_network_and_optimizers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a control of SPARKS on a pseudo-mouse watching Allen '
                                                 'natural movie one')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=1, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=9, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=16, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=32, help='Size of attention embeddings')
    parser.add_argument('--bottleneck_dim', type=int, default=8, help='Size of the bottleneck layer')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--tau_p', type=int, default=6, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, nargs='+', action=FloatOrList, default=5.,
                        help='STDP decay, or several (space-separated) for one head each')
    parser.add_argument('--kl_type', type=str, default='sequential', choices=['sequential', 'standard'],
                        help='Type of KL divergence to use')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing for cross-entropy loss')
    parser.add_argument('--chunk_length', type=int, default=None, help='Length of each segment')
    parser.add_argument('--block_type', type=str, default='symmetric', choices=list(CONTROL_BLOCKS),
                        help='The module that replaces the Hebbian attention block')

    # Data parameters
    parser.add_argument('--block', type=str, default='first',
                        choices=['first', 'second', 'across', 'both'], help='From which blocks to use')
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prediction', 'reconstruction', 'unsupervised'],
                        help='Which type of task to perform')
    parser.add_argument('--n_neurons', type=int, default=50)
    parser.add_argument('--dt', type=float, default=0.005, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed of every random number generator; default None (unseeded)')

    args = parser.parse_args()
    set_seed(args.seed)

    # Create folder to save results
    make_res_folder('allen_movies_pseudomouse_' + args.mode + '_nneurons_' + str(args.n_neurons),
                    os.path.join(os.getcwd(), f'results/allen/movies/{args.mode}/{args.block_type}/'
                                              f'block/{args.block}/n_neurons/{args.n_neurons}'), args)

    # Scale the learning rate with the number of neurons (after make_res_folder, so commandline_args.txt
    # keeps the value passed on the command line)
    args.lr = args.lr / np.sqrt(args.n_neurons)
    neuron_types = ['VISp']
    frames = get_frames(os.path.join(args.home, "datasets/allen_visual/"), dt=args.dt, ds=args.ds)
    (train_dataset,
     test_dataset,
     train_dl,
     test_dl) = make_pseudomouse_movies_data(os.path.join(args.home, "datasets/allen_visual/"),
                                             areas=neuron_types,
                                             n_neurons=args.n_neurons,
                                             dt=args.dt,
                                             block=args.block,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             # Reconstruction uses the same trials as prediction;
                                             # get_targets builds its frame targets
                                             mode='prediction' if args.mode == 'reconstruction' else args.mode,
                                             seed=args.seed)  # seeds the random draw of units

    np.save(args.results_path + '/good_units_ids.npy', train_dataset.good_units_ids)

    model, optimizer, scheduler, loss_fn, kl_transition = make_network_and_optimizers(args,
                                                                                      datasets=[train_dataset],
                                                                                      frames=frames)
    best_test_acc = -np.inf
    # The decoder outputs logits; the modes trained with binary cross-entropy are reported as probabilities
    act = torch.sigmoid if args.mode in ['reconstruction', 'unsupervised'] else None

    # Training loop
    pbar = tqdm.tqdm(range(args.n_epochs))
    for epoch in pbar:
        for inputs, targets in train_dl:
            targets = get_targets(targets, mode=args.mode, frames=frames, dt=args.dt)
            train_on_batch(model=model,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           beta=args.beta,
                           transition=kl_transition,
                           chunk_length=args.chunk_length)

            scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            # Unused, but creating a DataLoader iterator draws a seed from torch's global RNG:
            # it is kept so that the random number stream, and thus seeded results, are unchanged
            test_iterator = iter(test_dl)
            test_loss = 0
            encoder_outputs = torch.Tensor()
            decoder_outputs = torch.Tensor()

            test_iterator = iter(test_dl)
            for inputs, targets in test_iterator:
                targets = get_targets(targets, mode=args.mode, frames=frames, dt=args.dt)
                loss, encoder_outputs_batch, decoder_outputs_batch = evaluate_on_batch(model=model,
                                                                                       inputs=inputs,
                                                                                       targets=targets,
                                                                                       loss_fn=loss_fn,
                                                                                       act=act,
                                                                                       chunk_length=args.chunk_length)
                test_loss = test_loss + loss
                encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch.cpu()), dim=0)
                decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch.cpu()), dim=0)

            # Model selection maximises the frame accuracy, or minimises the test loss for the other modes
            if args.mode == 'prediction':
                test_acc = calculate_test_acc(decoder_outputs, dt=args.dt)
            else:
                test_acc = -test_loss

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc,
                                         encoder_outputs, decoder_outputs, model)

            if test_acc == 1. and args.mode == 'prediction':
                print("Perfect test accuracy reached, stopping training.")
                break

            if args.mode == 'prediction':
                pbar.set_description("Epoch %d, test acc: %.3f" % (epoch, test_acc))
            else:
                pbar.set_description("Epoch %d, test loss: %.3f" % (epoch, -test_acc))
