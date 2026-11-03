# sparks-controls

The control models of *A Biologically Inspired Attention Model for Neural Signal Analysis*
(Skatchkovsky et al.): [SPARKS](https://github.com/FrancisCrickInstitute/SPARKS) with its Hebbian attention
block replaced by other modules, to test what the Hebbian attention contributes. Everything else in the model
(the shared blocks, the latent space, the decoder, the training) is SPARKS' own.

## Control modules

The `controls` package defines them; the scripts select one with `--block_type`.

| `--block_type` | Module | What replaces the Hebbian attention block |
| --- | --- | --- |
| `symmetric` | `SymmetricHebbianAttentionBlock` | Each neuron is embedded from a symmetric, unbounded Hebbian coefficient with every other neuron, which grows with co-activation whichever neuron fires first |
| `correlation` | `CorrelationAttentionBlock` | Each neuron is embedded from the running correlation of its smoothed firing rate with every other neuron's, with no learned dynamics |
| `mlp` | `MLPControlBlock` | An MLP at each timestep over every neuron's smoothed firing rate (an exponential moving average) |
| `gru` | `GRUControlBlock` | A GRU over time over every neuron's smoothed firing rate |
| `transformer` | `TransformerControlBlock` | A causal self-attention transformer block over time over every neuron's smoothed firing rate |

The coefficient controls (`symmetric`, `correlation`) keep the rest of the Hebbian attention block: one embedding
of each neuron per STDP time constant (`tau_s`), mixed by the same projection, then the same feed-forward network
and Perceiver readout, which compresses the neurons into the latent stream. Only the spike-timing coefficients of
each pair of neurons are replaced by other ones. The sequence controls (`mlp`, `gru`, `transformer`) replace both
the Hebbian attention block and the Perceiver: the smoothed firing rates of every neuron, one per `tau_s`, are read
out by a per-session linear layer, then the sequence model writes directly to the latent stream.

## Installation

```bash
pip install -r requirements.txt
```

The control modules build on internals of sparks (`sparks.models.front_end`, `sparks.models.scan`), so keep
sparks-ai at the version that `requirements.txt` pins.

## Usage

Run the scripts from the repository root; `--home` is the folder that contains the `datasets/` folder (the
same data as for [sparks-examples](https://github.com/Iacaruso-lab/sparks-examples)):

| Script | Data | Paper |
| --- | --- | --- |
| `python -m scripts.mec.train_mec` | Mouse MEC calcium imaging: co-smoothing, unsupervised | Figures 2E, 2G, S2E-J |
| `python -m scripts.monkey.area2_bump` | NLB Area2_Bump: behaviour decoding, co-smoothing, unsupervised | Figure S3 |
| `python -m scripts.allen_visual.train_allen_movies_pseudomouse` | Allen Visual Coding, natural movie one: frame reconstruction and decoding | Figure S4A |

For example:

```bash
python -m scripts.mec.train_mec --home=/path/to/data --mode=co-smoothing --block_type=gru --tau_p=1 --tau_f=1 \
    --lr=0.005 --n_epochs=5000 --seed=1
```

Each script lists its options with `--help`, and saves its results under `results/` in the current directory.

The commands of every control run in the paper are in `paper_configurations.sh`, one section per figure: list
them with `bash paper_configurations.sh`, print a section's commands with
`DRY_RUN=1 bash paper_configurations.sh fig2_mec_controls`, and run it with
`DATA_HOME=/path/to/data bash paper_configurations.sh fig2_mec_controls`. The runs of SPARKS itself are in the
`paper_configurations.sh` of sparks-examples.

## Tests

```bash
python -m pytest
```

The tests check that each control plugs into SPARKS, carries its state across chunks, and still produces
the reference outputs of `tests/golden_outputs.pt`.

## Citation

If you use this code, please cite:

```bibtex
@article{skatchkovsky2024sparks,
  title   = {A Biologically Inspired Attention Model for Neural Signal Analysis},
  author  = {Skatchkovsky, Nicolas and Glazman, Natalia and Egea-Weiss, Alexander and Sadeh, Sadra and Iacaruso, Florencia},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.08.13.607787}
}
```

## License

This code is free software, released under the [GNU General Public License v3.0 or later](LICENSE.md).
