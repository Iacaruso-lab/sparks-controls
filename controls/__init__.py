"""
The control modules of the paper's ablations, which replace SPARKS' Hebbian attention block (each is a
sparks.models.FrontEndBlock, selected with HebbianAttentionConfig(block_class=...)):

- SymmetricHebbianAttentionBlock and CorrelationAttentionBlock embed each neuron, like the Hebbian attention block,
  from a coefficient with every other neuron (a symmetric Hebbian rule, a correlation) instead of STDP, then mix
  the tau_s heads and compress the neurons with the same Perceiver readout
  (sparks.models.front_end.PerceiverFrontEnd), so that only the neurons' embeddings differ;
- MLPControlBlock, GRUControlBlock and TransformerControlBlock replace both the Hebbian attention block and the
  Perceiver: a per-session linear readout of the exponential moving averages of every neuron's spikes, followed by
  a sequence model shared by every session that writes directly to the latent stream (see
  base.EMAControlBlockBase).

They build on internals of sparks (sparks.models.front_end, sparks.models.scan), hence the pinned version of
sparks-ai in requirements.txt.
"""
from controls.coefficient_blocks import (
    SymmetricHebbianAttentionBlock,
    CorrelationAttentionBlock,
)
from controls.sequence_blocks import (
    MLPControlBlock,
    GRUControlBlock,
    TransformerControlBlock,
)

__all__ = [
    'SymmetricHebbianAttentionBlock',
    'CorrelationAttentionBlock',
    'MLPControlBlock',
    'GRUControlBlock',
    'TransformerControlBlock',
]
