"""Basic operator constructors."""

from __future__ import annotations
import numpy as np

def destroy(dim: int) -> np.ndarray:
    """annihilation operator for a dim-dimensional harmonic mode."""
    if dim < 1:
        raise ValueError("dim must be >= 1")
    op = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        op[n - 1, n] = np.sqrt(n)
    return op

def create(dim: int) -> np.ndarray:
    """the creation operator."""
    return destroy(dim).conj().T

def number(dim: int) -> np.ndarray:
    """the number operator."""
    a = destroy(dim)
    return a.conj().T @ a

def identity(dim: int) -> np.ndarray:
    """the identity matrix."""
    if dim < 1:
        raise ValueError("dim must be >= 1")
    return np.eye(dim, dtype=complex)

def basis_vector(dim: int, index: int) -> np.ndarray:
    """a computational basis vector."""
    if not (0 <= index < dim):
        raise IndexError(f"index {index} out of range for dimension {dim}")
    vec = np.zeros(dim, dtype=complex)
    vec[index] = 1.0
    return vec
