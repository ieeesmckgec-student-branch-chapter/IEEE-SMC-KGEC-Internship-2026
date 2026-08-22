"""Exact statevector simulators for the two QAOA arms, without Qiskit in the loop.

Both ansaetze have the same structure: a diagonal cost unitary followed by a mixer
that factorises over disjoint qubit groups. That makes exact simulation cheap if we
exploit the structure instead of calling a general-purpose simulator.

`SubspaceXYSimulator`
    XY mixing never leaves the stage-wise one-hot subspace, so the state lives in
    k**s dimensions rather than 2**(s*k). For 5 stages x 5 candidates that is 3125
    amplitudes instead of 33.5 million. Within a block, exp[-i b (XX+YY)/2] acts on
    the Hamming-weight-1 sector as a two-level rotation, so the whole block mixer is
    a k x k matrix and the layer mixer is a tensor product of s of them.

`FullSpaceSimulator`
    Used for the penalty + X-mixer baseline, which genuinely needs all 2**n states.
    Still much faster than a gate-by-gate simulator because the cost layer is a
    single elementwise multiply and the mixer is one 2x2 contraction per qubit.

Both are validated against Qiskit Aer in `scripts/03_validate_fast_sim.py`.
"""

from __future__ import annotations

import numpy as np

try:  # optional GPU backend
    import cupy
except ImportError:  # pragma: no cover - GPU is optional
    cupy = None


def _array_module(device: str):
    if device == "gpu":
        if cupy is None:
            raise RuntimeError("device='gpu' requires cupy")
        return cupy
    return np


def ring_pairs(k: int) -> list[tuple[int, int]]:
    """Mixer pair ordering, matching `agritrust_core.apply_xy_mixer_layer`."""
    if k < 2:
        return []
    if k == 2:
        return [(0, 1)]
    return [(q, (q + 1) % k) for q in range(k)]


def xy_block_mixer(k: int, beta: float) -> np.ndarray:
    """k x k unitary of one XY mixer layer restricted to the one-hot sector.

    Each `RXX(beta) RYY(beta)` pair implements exp[-i beta (X_iX_j + Y_iY_j)/2],
    which on span{|..1_i..0_j..>, |..0_i..1_j..>} is
        [[cos beta, -i sin beta], [-i sin beta, cos beta]]
    and acts as the identity on one-hot states whose excitation sits elsewhere.
    Gates compose left-to-right in circuit order, so the matrices multiply in
    reverse application order.
    """
    matrix = np.eye(k, dtype=complex)
    cos, sin = np.cos(beta), -1j * np.sin(beta)
    for i, j in ring_pairs(k):
        pair = np.eye(k, dtype=complex)
        pair[i, i] = cos
        pair[j, j] = cos
        pair[i, j] = sin
        pair[j, i] = sin
        matrix = pair @ matrix
    return matrix


def rx_matrix(beta: float) -> np.ndarray:
    """Qiskit RX(2*beta), the single-qubit X-mixer factor."""
    cos, sin = np.cos(beta), -1j * np.sin(beta)
    return np.array([[cos, sin], [sin, cos]], dtype=complex)


def _apply_along_axis(state: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(state, axis, -1)
    contracted = moved @ matrix.T
    return np.moveaxis(contracted, -1, axis)


def _apply_single_qubit(state, matrix, qubit: int, xp):
    """Apply a 2x2 gate to `qubit` of a flat 2**n statevector, in place.

    Reshaping to (high, 2, low) and slicing touches each amplitude exactly twice.
    The generic moveaxis-and-matmul path copies the whole array several times per
    qubit, which at 2**25 amplitudes is the difference between 64 s and 0.2 s.
    """
    stride = 1 << qubit
    view = state.reshape(-1, 2, stride)
    lower = view[:, 0, :]
    upper = view[:, 1, :]
    new_lower = matrix[0, 0] * lower + matrix[0, 1] * upper
    upper *= matrix[1, 1]
    upper += matrix[1, 0] * lower
    view[:, 0, :] = new_lower
    return state


class SubspaceXYSimulator:
    """Exact XY-QAOA in the stage-wise one-hot subspace.

    State is held as an array of shape (k,)*s where index (c_0, ..., c_{s-1}) is the
    route that picks candidate c_i at stage i.
    """

    def __init__(self, feasible_costs: np.ndarray, n_stages: int, candidates_per_stage: int):
        self.n_stages = n_stages
        self.k = candidates_per_stage
        self.shape = (candidates_per_stage,) * n_stages
        self.costs = np.asarray(feasible_costs, dtype=float).reshape(self.shape)
        self.dimension = int(np.prod(self.shape))

    def initial_state(self) -> np.ndarray:
        """Tensor product of W states = uniform superposition over feasible routes."""
        return np.full(self.shape, 1.0 / np.sqrt(self.dimension), dtype=complex)

    def run(self, gammas, betas) -> np.ndarray:
        state = self.initial_state()
        for gamma, beta in zip(gammas, betas):
            state = state * np.exp(-1j * gamma * self.costs)
            mixer = xy_block_mixer(self.k, beta)
            for axis in range(self.n_stages):
                state = _apply_along_axis(state, mixer, axis)
        return state

    def probabilities(self, gammas, betas) -> np.ndarray:
        """Probabilities over feasible routes, flattened in row-major (c_0 slowest)."""
        return (np.abs(self.run(gammas, betas)) ** 2).ravel()


class FullSpaceSimulator:
    """Exact QAOA over all 2**n basis states with a product X mixer.

    Set `device="gpu"` to run on CuPy. At 25 qubits the state is 536 MB in
    complex128 or 268 MB in complex64, and the whole simulation is memory-bandwidth
    bound, so a GPU is roughly two orders of magnitude faster.
    """

    def __init__(self, basis_costs, n_qubits: int, device: str = "cpu", precision="complex128"):
        self.n_qubits = n_qubits
        self.device = device
        self.precision = precision
        self.xp = _array_module(device)
        real = "float64" if precision == "complex128" else "float32"
        self.costs = self.xp.asarray(np.asarray(basis_costs, dtype=float).ravel(), dtype=real)
        self._cost_order = None

    def initial_state(self):
        amplitude = 2.0 ** (-self.n_qubits / 2)
        return self.xp.full(1 << self.n_qubits, amplitude, dtype=self.precision)

    def run(self, gammas, betas):
        xp = self.xp
        state = self.initial_state()
        for gamma, beta in zip(gammas, betas):
            state *= xp.exp((-1j * float(gamma)) * self.costs).astype(self.precision)
            mixer = np.asarray(rx_matrix(beta), dtype=self.precision)
            for qubit in range(self.n_qubits):
                state = _apply_single_qubit(state, mixer, qubit, xp)
        return state

    def probabilities(self, gammas, betas):
        state = self.run(gammas, betas)
        probabilities = self.xp.abs(state) ** 2
        if self.device == "gpu":
            return probabilities.get()
        return probabilities

    def objective(self, gammas, betas, kind: str = "expectation", alpha: float = 0.25) -> float:
        """Evaluate an objective on the simulation device and return one scalar.

        Keeping the probability vector on-device avoids copying 2**n values over
        PCIe for every optimizer evaluation. The CVaR ordering is cached because
        costs are fixed throughout an optimization run.
        """
        xp = self.xp
        state = self.run(gammas, betas)
        probabilities = xp.abs(state) ** 2

        if kind == "expectation":
            return float(xp.dot(probabilities, self.costs).item())
        if kind != "cvar":
            raise ValueError(f"Unknown objective: {kind}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("CVaR alpha must be in (0, 1].")

        if self._cost_order is None:
            self._cost_order = xp.argsort(self.costs)
        sorted_costs = self.costs[self._cost_order]
        sorted_probabilities = probabilities[self._cost_order]
        cumulative = xp.cumsum(sorted_probabilities)
        cutoff = int(xp.searchsorted(cumulative, alpha, side="left").item())
        cutoff = min(cutoff, len(sorted_costs) - 1)
        weights = sorted_probabilities[: cutoff + 1].copy()
        weights[-1] -= cumulative[cutoff] - alpha
        total = weights.sum()
        if float(total.item()) <= 0.0:
            return float(sorted_costs[0].item())
        return float(xp.dot(weights, sorted_costs[: cutoff + 1]).item() / total.item())


def feasible_index_map(n_stages: int, candidates_per_stage: int) -> np.ndarray:
    """Flat full-space index of each feasible route, in SubspaceXYSimulator order.

    Route (c_0, ..., c_{s-1}) sets qubit `i*k + c_i`, so its little-endian integer is
    sum_i 2**(i*k + c_i).
    """
    k = candidates_per_stage
    indices = np.zeros((candidates_per_stage,) * n_stages, dtype=np.int64)
    for flat, choices in enumerate(np.ndindex(*((k,) * n_stages))):
        indices.flat[flat] = sum(1 << (stage * k + choice) for stage, choice in enumerate(choices))
    return indices.ravel()


def feasible_costs_from_qubo(linear, quadratic, n_stages: int, candidates_per_stage: int):
    """Cost of every feasible route, in SubspaceXYSimulator (row-major) order."""
    k = candidates_per_stage
    linear = np.asarray(linear, dtype=float)
    costs = np.zeros((k,) * n_stages, dtype=float)
    for flat, choices in enumerate(np.ndindex(*((k,) * n_stages))):
        active = [stage * k + choice for stage, choice in enumerate(choices)]
        value = float(linear[active].sum())
        active_set = set(active)
        for (i, j), coefficient in quadratic.items():
            if i in active_set and j in active_set:
                value += coefficient
        costs.flat[flat] = value
    return costs.ravel()
