"""
Baselines Package for Long-Range Temporal Alignment Research.

Central repository of benchmark implementations, control configurations,
evaluation engines, and baseline reproduction notebooks:
- ticktack: Sinusoidal polar temporal encoding (Han et al., 2025) & control matrix (C1-C4).
- fone: Fourier Neural Embeddings for numerical years (LOGGING 102).
- discrete: Learned discrete Can Chi embeddings and text tag prompting baselines.
"""

from baselines import discrete, fone, ticktack

__version__ = "0.1.0"
__all__ = ["ticktack", "fone", "discrete"]
