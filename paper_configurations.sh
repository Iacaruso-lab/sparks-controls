#!/usr/bin/env bash
# Configurations of the control runs behind the figures of
#
#   N. Skatchkovsky, N. Glazman, A. Egea-Weiss, S. Sadeh and F. Iacaruso,
#   "A Biologically Inspired Attention Model for Neural Signal Analysis":
#
# SPARKS with its Hebbian attention block replaced by other modules (--block_type, see the controls package). The
# runs of SPARKS itself are in paper_configurations.sh of the sparks-examples repository.

# Usage:
#   bash paper_configurations.sh                        lists the sections
#   bash paper_configurations.sh SECTION [SECTION ...]  runs the sections, one job after the other
#   DRY_RUN=1 bash paper_configurations.sh SECTION      prints the commands instead of running them
#
# Environment variables:
#   DATA_HOME        folder that contains datasets/, passed to every training script as --home:
#                      datasets/mec/calcium_activity_matrix_60584_session17.mat          (Figure 2)
#                      datasets/000127/sub-Han/sub-Han_desc-train_behavior+ecephys.nwb   (Figure S3, DANDI 000127)
#                      datasets/allen_visual/                                             (Figure S4, see the
#                                                                                         allen_preprocessing section
#                                                                                         of sparks-examples)
#   OUT_ROOT         where the results go (default: ./paper_runs). Each section runs in its own folder,
#                    OUT_ROOT/<section>/, under which the scripts write results/... with the same layout as the
#                    published results.
#

set -euo pipefail

CONTROLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-$PWD/paper_runs}"
DATA_HOME="${DATA_HOME:-}"
SECTION_DIR=""

# Runs a command from the current section's folder, with this repository importable; prints it with DRY_RUN=1
run() {
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "$*"
    else
        (cd "$SECTION_DIR" && PYTHONPATH="$CONTROLS_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$@")
    fi
}

# Runs a module of this repository: py <module> <arguments>
py() {
    run python -m "$@"
}


# ======================================================================================================================
# Figure 2 and Figure S2: medial entorhinal cortex (MEC), calcium imaging, one session of 484 neurons
# ======================================================================================================================
# Every model is trained for co-smoothing: 10% of the neurons, drawn from --seed, are left out and predicted from the
# others. The models are trained on 600-2600 s of the session and tested on 2600-3600 s (--last_stop=2600).

## fig2_mec_controls: Hebbian attention replaced by other modules (2E, 2G, S2E-J)
# 5 modules x 16 cells x 20 runs (seeds 1-20), plus seeds 21-30 in the best cell of each module, which S2H-J
# average too (S2F has 20 runs per module). 2E and S2E show the symmetric rule's (1, 1) run with seed 5, 2G one run
# per cell of the symmetric rule. In co-smoothing, the script sets tau_s = 10, 50, 100 and 500.
fig2_mec_controls() {
    local args=(--home="$DATA_HOME" --mode=co-smoothing --n_epochs=5000 --test_period=5 --lr=0.005 --beta=0.0001
                --batch_size=1 --latent_dim=2 --bottleneck_dim=8 --embed_dim=32 --n_heads=1 --n_layers=0 --cd_rate=0.4
                --chunk_length=1000 --last_stop=2600 --p_left_out=0.1 --ds=4 --kl_type=sequential --patience=500)
    local block tau_p tau_f seed
    for block in symmetric mlp gru transformer correlation; do
        for tau_p in 1 2 5 10; do
            for tau_f in 1 2 5 10; do
                for seed in $(seq 1 20); do
                    py scripts.mec.train_mec "${args[@]}" --tau_p=$tau_p --tau_f=$tau_f --block_type=$block \
                        --seed=$seed
                done
            done
        done
    done
    for seed in $(seq 21 30); do
        for block in mlp gru transformer correlation; do
            py scripts.mec.train_mec "${args[@]}" --tau_p=1 --tau_f=1 --block_type=$block --seed=$seed
        done
        py scripts.mec.train_mec "${args[@]}" --tau_p=10 --tau_f=10 --block_type=symmetric --seed=$seed
    done
}


# ======================================================================================================================
# Figure S3: Area2_Bump (Neural Latents Benchmark, DANDI 000127), monkey reaching
# ======================================================================================================================
# Spikes of 65 neurons of area 2, 1 ms bins, from 100 ms before to 500 ms after movement onset.

## fig3_area2_controls: Hebbian attention replaced by other modules (S3A-E, S3H, S3I)
# Supervised (S3A-E, S3H): 10 runs per module (19 for correlation).
# Co-smoothing (S3I): 10 runs per module; the script sets tau_s = 0.01, 0.05, 0.1 and 0.5 s in this mode.
fig3_area2_controls() {
    local args=(--home="$DATA_HOME" --test_period=1 --batch_size=64 --beta=0.0001 --embed_dim=32
                --n_heads=4 --dropout=0 --kl_type=sequential --patience=200 --eps=0.001)
    local block n_runs seed
    for block in symmetric mlp gru transformer correlation; do
        for seed in $(seq 1 10); do
            py scripts.monkey.area2_bump "${args[@]}" --block_type=$block --mode=prediction --target_type=hand_pos \
                --n_epochs=1000 --lr=0.01 --latent_dim=2 --bottleneck_dim=2 --n_layers=6 --tau_p=10 --tau_f=1 \
                --tau_s=0.5 --cd_rate=0 --seed=$seed
        done
        for seed in $(seq 1 10); do
            py scripts.monkey.area2_bump "${args[@]}" --block_type=$block --mode=co-smoothing --n_epochs=2000 \
                --lr=0.005 --cd_rate=0.4 --latent_dim=60 --bottleneck_dim=16 --n_layers=0 --tau_p=5 --tau_f=15 \
                --seed=$seed
        done
    done
}


# ======================================================================================================================
# Figure S4: Allen Visual Coding Neuropixels, natural movie one
# ======================================================================================================================
# A pseudo-mouse of VISp units sampled at random from all the sessions; 30 s movie (900 frames, spatially downsampled
# by 2), 5 ms bins, 10 presentations per block.

## fig4_allen_controls: Hebbian attention replaced by other modules (S4A)
# 5 modules x 10 runs, 100 neurons. S4A compares them with the 100-neuron runs of the fig4_allen_reconstruction
# section of sparks-examples.
fig4_allen_controls() {
    local block seed
    for block in symmetric correlation transformer mlp gru; do
        for seed in $(seq 1 10); do
            py scripts.allen_visual.train_allen_movies_pseudomouse --home="$DATA_HOME" --test_period=1 --n_heads=1 \
                --tau_p=6 --tau_f=1 --kl_type=sequential --dropout=0 --label_smoothing=0.1 --dt=0.005 --ds=2 \
                --mode=reconstruction --block=first --n_neurons=100 --block_type=$block --n_epochs=500 \
                --batch_size=5 --lr=0.01 --beta=0.1 --embed_dim=32 --bottleneck_dim=8 --latent_dim=64 \
                --n_layers=0 --chunk_length=1000 --tau_s=5. --seed=$seed
        done
    done
}


# ======================================================================================================================

main() {
    local this_file="${BASH_SOURCE[0]}" section
    if (( $# == 0 )); then
        echo "Usage: [DRY_RUN=1] DATA_HOME=... [OUT_ROOT=...] bash $(basename "$this_file") SECTION [SECTION ...]"
        echo "Sections:"
        grep -E '^## ' "$this_file" | sed 's/^## /  /'
        return 0
    fi
    for section in "$@"; do
        if ! grep -qE "^## $section:" "$this_file"; then
            echo "Unknown section: $section" >&2
            return 1
        fi
    done
    if [[ -n "${DRY_RUN:-}" ]]; then
        DATA_HOME="${DATA_HOME:-\$DATA_HOME}"
    else
        : "${DATA_HOME:?Set DATA_HOME to the folder that contains datasets/}"
    fi
    for section in "$@"; do
        SECTION_DIR="$OUT_ROOT/$section"
        if [[ -z "${DRY_RUN:-}" ]]; then
            mkdir -p "$SECTION_DIR"
        fi
        echo "# $section (in $SECTION_DIR)"
        "$section"
    done
}

main "$@"
