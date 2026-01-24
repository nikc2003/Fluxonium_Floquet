from __future__ import annotations

from cProfile import label
import itertools
from itertools import chain, product

import numpy as np
import qutip as qt

from .utils.file_io import Serializable


class Model(Serializable):
    """Specify the model, including the Hamiltonian, drive strengths and frequencies.

    Can be subclassed to e.g. override the hamiltonian() method for a different (but
    still periodic!) Hamiltonian.

    Parameters:
        H0: Drift Hamiltonian, which must be diagonal and provided in units such that
            H0 can be passed directly to qutip.
        H1: Drive operator, which should be unitless (for instance the charge-number
            operator n of the transmon). It will be multiplied by a drive amplitude
            that we scan over from drive_parameters.drive_amplitudes.
        omega_d_values: drive frequencies to scan over
        drive_amplitudes: amp values to scan over. Can be one dimensional in which case
            these amplitudes are used for all omega_d, or it can be two dimensional
            in which case the first dimension are the amplitudes to scan over
            and the second are the amplitudes for respective drive frequencies

    """

    def __init__(
        self,
        H0: qt.Qobj | np.ndarray | list,
        H1: qt.Qobj | np.ndarray | list,
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
    ):
        if not isinstance(H0, qt.Qobj):
            H0 = qt.Qobj(np.array(H0, dtype=complex))
        if not isinstance(H1, qt.Qobj):
            H1 = qt.Qobj(np.array(H1, dtype=complex))
        if isinstance(omega_d_values, list):
            omega_d_values = np.array(omega_d_values)
        if isinstance(drive_amplitudes, list):
            drive_amplitudes = np.array(drive_amplitudes)
        if len(drive_amplitudes.shape) == 1:
            drive_amplitudes = np.tile(drive_amplitudes, (len(omega_d_values), 1)).T
        assert len(drive_amplitudes.shape) == 2
        assert drive_amplitudes.shape[1] == len(omega_d_values)

        self.H0 = H0
        self.H1 = H1
        self.omega_d_values = omega_d_values
        self.drive_amplitudes = drive_amplitudes

    def omega_d_to_idx(self, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to omega_d value."""
        return np.argmin(np.abs(self.omega_d_values - omega_d))

    def amp_to_idx(self, amp: float, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to amplitude value.

        Because the drive amplitude can depend on the drive frequency, we also must pass
        the drive frequency here.
        """
        omega_d_idx = self.omega_d_to_idx(omega_d)
        return np.argmin(np.abs(self.drive_amplitudes[:, omega_d_idx] - amp))

    def omega_d_amp_params(self, amp_idxs: list) -> itertools.chain:
        """Return ordered chain object of the specified omega_d and amplitude values."""
        amp_range_vals = self.drive_amplitudes[amp_idxs[0] : amp_idxs[1]]
        _omega_d_amp_params = [
            product([omega_d], amp_vals)
            for omega_d, amp_vals in zip(
                self.omega_d_values, amp_range_vals.T, strict=False
            )
        ]
        return chain(*_omega_d_amp_params)

    def hamiltonian(self, omega_d_amp: tuple[float, float]) -> list[qt.Qobj]:
        """Return the Hamiltonian we actually simulate."""
        omega_d, amp = omega_d_amp
        return qt.QobjEvo([self.H0, [amp * self.H1, lambda t: np.cos(omega_d * t)]])

from typing import Iterable, Tuple, Sequence, Union, Optional


def _as_qobj(x):
    """Convert input to Qobj if not already."""
    return x if isinstance(x, qt.Qobj) else qt.Qobj(np.array(x, dtype=complex))


def _diag_unitary(H: qt.Qobj) -> tuple:
    """Return diagonal unitary from eigenvalues of H."""
    evals, evecs = H.eigenstates()
    U_full = np.column_stack([evec.full().flatten() for evec in evecs])
    U = qt.Qobj(U_full, dims=H.dims)
    H0_diag = qt.Qobj(np.diag(np.real(evals)), dims=H.dims)
    return H0_diag, U, np.array(evals), evecs


def _dress(bare_op: qt.Qobj, U: qt.Qobj) -> qt.Qobj:
    """Return the dressed operator U^dagger bare_op U."""
    if not isinstance(bare_op, qt.Qobj):
        bare_op = qt.Qobj(np.array(bare_op, dtype=complex), dims=U.dims)
    elif bare_op.dims != U.dims:
        bare_op = qt.Qobj(bare_op.full(), dims=U.dims)
    return U.dag() * bare_op * U


class CompositeModel(Model):
    def __init__(
        self,
        H0_bare: qt.Qobj | np.ndarray | list,
        H1_drive: qt.Qobj | np.ndarray | list | list[qt.Qobj],
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
        subsystem_dims: tuple[int, int],
        drive_weights: list[complex] | None = None,
        state_labels_dims: tuple[int, int] | None = None
    ):
        H_bare = _as_qobj(H0_bare)
        self.H0_bare = H_bare
        self.H1_drive = H1_drive
        self.subsystem_dims = subsystem_dims
        self.drive_weights = drive_weights
        self.state_labels_dims = state_labels_dims or subsystem_dims
        self._label_to_index_cache = {}

        # diagonalize the bare hamiltonian
        H0_diag, U, evals, evecs = _diag_unitary(H_bare)

        # dress the drive operator
        if isinstance(H1_drive, (list, tuple)):
            if drive_weights is None:
                drive_weights = [1.0] * len(H1_drive)
            H1_tot = sum(
                _as_qobj(H1) * w for H1, w in zip(H1_drive, drive_weights, strict=False)
            )
        else:
            H1_tot = _as_qobj(H1_drive)

        H1_dressed = _dress(H1_tot, U)

        super().__init__(
            H0=H0_diag,
            H1=H1_dressed,
            omega_d_values=omega_d_values,
            drive_amplitudes=drive_amplitudes,
        )

        self.U = U
        self.evals = evals
        self.evecs = evecs

    def label_to_index(self, label: tuple[int, int]) -> int:
        if label in self._label_to_index_cache:
            return self._label_to_index_cache[label]
        Nq, Nr = self.state_labels_dims
        nq, nr = label
        bare_ket = qt.tensor(qt.basis(Nq, nq), qt.basis(Nr, nr))
        coeffs = (self.U.dag() * bare_ket).full().flatten()
        idx = int(np.argmax(np.abs(coeffs)))
        self._label_to_index_cache[label] = idx
        max_overlap = np.max(np.abs(coeffs))
        if max_overlap < 0.7:
            print(f"Warning: label {label} has poor overlap with any eigenstate "
                f"(max |overlap| = {max_overlap:.3f}). Mapping may be ambiguous.")
        return idx

    def labels_to_indices(self, labels: list[tuple[int, int]]) -> list:
        """Convert a list of tuple labels to flat indices."""
        return [self.label_to_index(label) for label in labels]