"""Noise and symmetry-post-selection tools for constrained XY-QAOA.

The scalable channel in this module is an exact *end-of-circuit measurement
proxy*: it applies independent classical 0->1 and 1->0 transitions to an ideal
computational-basis distribution.  It captures the measurement-equivalent X/Y
part of local depolarization and amplitude damping immediately before
measurement.  It does not model noise accumulating through the QAOA gates,
coherent errors, crosstalk, or topology-dependent routing.

The separate Aer helpers apply genuine gate-level depolarizing and thermal
relaxation errors to circuits transpiled to ``rz/sx/x/cx``.  They are intended
for small validation cases, not the 25-qubit scaling sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)

HARDWARE_BASIS_GATES = ("rz", "sx", "x", "cx")


@dataclass(frozen=True)
class LocalNoiseParameters:
    """Independent local end-of-circuit noise strengths.

    Amplitude damping is applied first, followed by a symmetric bit flip.
    ``bit_flip`` is the measurement-equivalent X/Y transition probability, not
    Qiskit's depolarizing-channel parameter.
    """

    bit_flip: float = 0.0
    amplitude_damping: float = 0.0

    def __post_init__(self):
        for name, value in (
            ("bit_flip", self.bit_flip),
            ("amplitude_damping", self.amplitude_damping),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

    @property
    def zero_to_one(self) -> float:
        return float(self.bit_flip)

    @property
    def one_to_zero(self) -> float:
        damping = float(self.amplitude_damping)
        flip = float(self.bit_flip)
        return damping * (1.0 - flip) + (1.0 - damping) * flip


def _validated_distribution(probabilities, name: str) -> np.ndarray:
    distribution = np.asarray(probabilities, dtype=float)
    if distribution.ndim != 1 or distribution.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if distribution.size & (distribution.size - 1):
        raise ValueError(f"{name} length must be a power of two.")
    if not np.all(np.isfinite(distribution)):
        raise ValueError(f"{name} must contain finite probabilities.")
    if np.any(distribution < -1e-15):
        raise ValueError(f"{name} probabilities must be non-negative.")
    total = float(distribution.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must sum to one, got {total}.")
    return np.clip(distribution, 0.0, None)


def apply_independent_local_noise(
    probabilities: np.ndarray,
    noise: LocalNoiseParameters,
) -> np.ndarray:
    """Apply an exact tensor product of local measurement transitions.

    The implementation makes one streaming pass per qubit through the
    probability vector.  It costs O(n 2**n) time and O(2**n) output storage, but
    never constructs the O(4**n) transition matrix.
    """

    output = _validated_distribution(probabilities, "probabilities").copy()
    n_qubits = output.size.bit_length() - 1
    p_zero_to_one = noise.zero_to_one
    p_one_to_zero = noise.one_to_zero

    for qubit in range(n_qubits):
        stride = 1 << qubit
        pairs = output.reshape(-1, 2, stride)
        new_one = (
            p_zero_to_one * pairs[:, 0, :]
            + (1.0 - p_one_to_zero) * pairs[:, 1, :]
        )
        pairs[:, 0, :] += pairs[:, 1, :]
        pairs[:, 1, :] = new_one
        pairs[:, 0, :] -= new_one

    output[output < 0.0] = 0.0
    output /= output.sum()
    return output


def _feasible_indices(feasible, dimension: int) -> np.ndarray:
    values = np.asarray(feasible)
    if values.dtype == np.bool_:
        if values.ndim != 1 or values.size != dimension:
            raise ValueError("A feasible mask must match the full distribution length.")
        indices = np.flatnonzero(values)
    else:
        if values.ndim != 1:
            raise ValueError("Feasible indices must be one-dimensional.")
        indices = values.astype(np.int64)
        if not np.all(values == indices):
            raise ValueError("Feasible indices must be integers.")
        if np.any(indices < 0) or np.any(indices >= dimension):
            raise ValueError("A feasible index is outside the distribution.")
        if np.unique(indices).size != indices.size:
            raise ValueError("Feasible indices must not contain duplicates.")
    if indices.size == 0:
        raise ValueError("At least one feasible state is required.")
    return indices


def _metrics_from_feasible_probabilities(
    ideal_feasible: np.ndarray,
    noisy_feasible: np.ndarray,
    feasible_costs: np.ndarray,
    target_retained_shots: int,
) -> dict[str, float | bool]:
    if target_retained_shots <= 0:
        raise ValueError("target_retained_shots must be positive.")
    costs = np.asarray(feasible_costs, dtype=float)
    if costs.ndim != 1 or costs.shape != ideal_feasible.shape:
        raise ValueError("feasible_costs must align with the feasible states.")
    if not np.all(np.isfinite(costs)):
        raise ValueError("feasible_costs must be finite.")

    ideal_mass = float(ideal_feasible.sum())
    retained = float(noisy_feasible.sum())
    optimum_cost = float(np.min(costs))
    optimum_mask = np.isclose(costs, optimum_cost, rtol=1e-10, atol=1e-12)
    noisy_optimum_mass = float(noisy_feasible[optimum_mask].sum())

    base = {
        "ideal_feasible_probability": ideal_mass,
        "noisy_feasible_probability": retained,
        "retained_shot_fraction": retained,
        "noisy_probability_of_optimum": noisy_optimum_mass,
        "effective_shots_for_target": (
            float(target_retained_shots / retained) if retained > 0.0 else float("inf")
        ),
    }
    if retained <= 0.0:
        return {
            **base,
            "postselected_probability_of_optimum": 0.0,
            "conditional_mode_probability": 0.0,
            "conditional_mode_is_optimal": False,
            "conditional_mode_cost": float("nan"),
            "conditional_mode_cost_gap": float("nan"),
            "conditional_mode_approximation_ratio": float("nan"),
            "postselection_total_variation_distance": float("nan"),
        }

    conditional = noisy_feasible / retained
    mode_index = int(np.argmax(conditional))
    mode_cost = float(costs[mode_index])
    if mode_cost == 0.0:
        approximation_ratio = 1.0 if optimum_cost == 0.0 else float("nan")
    else:
        approximation_ratio = optimum_cost / mode_cost

    if ideal_mass > 0.0:
        ideal_conditional = ideal_feasible / ideal_mass
        tv_distance = 0.5 * float(np.abs(conditional - ideal_conditional).sum())
    else:
        tv_distance = float("nan")

    return {
        **base,
        "postselected_probability_of_optimum": noisy_optimum_mass / retained,
        "conditional_mode_probability": float(conditional[mode_index]),
        "conditional_mode_is_optimal": bool(optimum_mask[mode_index]),
        "conditional_mode_cost": mode_cost,
        "conditional_mode_cost_gap": mode_cost - optimum_cost,
        "conditional_mode_approximation_ratio": float(approximation_ratio),
        "postselection_total_variation_distance": tv_distance,
    }


def postselection_metrics(
    ideal_probabilities: np.ndarray,
    noisy_probabilities: np.ndarray,
    feasible,
    feasible_costs: np.ndarray,
    target_retained_shots: int = 1000,
) -> dict[str, float | bool]:
    """Measure feasibility loss, conditional quality, sampling cost, and bias."""

    ideal = _validated_distribution(ideal_probabilities, "ideal_probabilities")
    noisy = _validated_distribution(noisy_probabilities, "noisy_probabilities")
    if ideal.shape != noisy.shape:
        raise ValueError("Ideal and noisy distributions must have the same shape.")
    indices = _feasible_indices(feasible, ideal.size)
    return _metrics_from_feasible_probabilities(
        ideal[indices],
        noisy[indices],
        np.asarray(feasible_costs, dtype=float),
        target_retained_shots,
    )


def _expected_one_hot_indices(n_stages: int, candidates_per_stage: int) -> np.ndarray:
    shape = (candidates_per_stage,) * n_stages
    indices = np.empty(candidates_per_stage**n_stages, dtype=np.int64)
    for flat_index, choices in enumerate(np.ndindex(*shape)):
        indices[flat_index] = sum(
            1 << (stage * candidates_per_stage + choice)
            for stage, choice in enumerate(choices)
        )
    return indices


def _apply_matrix_along_axis(
    probabilities: np.ndarray,
    matrix: np.ndarray,
    axis: int,
) -> np.ndarray:
    moved = np.moveaxis(probabilities, axis, -1)
    transformed = moved @ matrix.T
    return np.moveaxis(transformed, -1, axis)


def one_hot_postselection_metrics(
    ideal_probabilities: np.ndarray,
    feasible_indices: np.ndarray,
    feasible_costs: np.ndarray,
    n_stages: int,
    candidates_per_stage: int,
    noise: LocalNoiseParameters,
    target_retained_shots: int = 1000,
) -> dict[str, float | bool]:
    """Exact post-selection metrics using only the one-hot output sector.

    This fast path is valid when the ideal input has support exclusively on the
    stage-wise one-hot sector and ``feasible_indices`` follows row-major route
    order.  It contracts a k x k retained-sector transition along each of the s
    route axes, reducing the sweep cost from O(n 2**n) to O(s k**(s+1)).
    """

    if n_stages <= 0 or candidates_per_stage <= 0:
        raise ValueError("n_stages and candidates_per_stage must be positive.")
    ideal = _validated_distribution(ideal_probabilities, "ideal_probabilities")
    n_qubits = n_stages * candidates_per_stage
    if ideal.size != 1 << n_qubits:
        raise ValueError("The ideal distribution length does not match the one-hot layout.")

    indices = _feasible_indices(feasible_indices, ideal.size)
    expected = _expected_one_hot_indices(n_stages, candidates_per_stage)
    if not np.array_equal(indices, expected):
        raise ValueError("feasible_indices must follow row-major one-hot route order.")
    ideal_feasible = ideal[indices]
    if not np.isclose(float(ideal_feasible.sum()), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("The one-hot fast path requires all ideal mass to be feasible.")

    k = candidates_per_stage
    p_zero_to_one = noise.zero_to_one
    p_one_to_zero = noise.one_to_zero
    retained_transition = np.full(
        (k, k),
        p_one_to_zero * p_zero_to_one * (1.0 - p_zero_to_one) ** max(k - 2, 0),
        dtype=float,
    )
    np.fill_diagonal(
        retained_transition,
        (1.0 - p_one_to_zero) * (1.0 - p_zero_to_one) ** (k - 1),
    )

    noisy_feasible = ideal_feasible.reshape((k,) * n_stages)
    for axis in range(n_stages):
        noisy_feasible = _apply_matrix_along_axis(
            noisy_feasible,
            retained_transition,
            axis,
        )
    return _metrics_from_feasible_probabilities(
        ideal_feasible,
        noisy_feasible.ravel(),
        np.asarray(feasible_costs, dtype=float),
        target_retained_shots,
    )


def build_gate_level_noise_model(
    depolarizing_1q: float,
    depolarizing_2q: float,
    t1_seconds: float,
    t2_seconds: float,
    one_qubit_gate_seconds: float = 35e-9,
    two_qubit_gate_seconds: float = 300e-9,
) -> NoiseModel:
    """Build hardware-basis depolarizing plus thermal-relaxation gate errors."""

    for name, value in (
        ("depolarizing_1q", depolarizing_1q),
        ("depolarizing_2q", depolarizing_2q),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if t1_seconds <= 0.0 or t2_seconds <= 0.0:
        raise ValueError("T1 and T2 must be positive.")
    if t2_seconds > 2.0 * t1_seconds:
        raise ValueError("T2 must not exceed 2*T1.")
    if one_qubit_gate_seconds < 0.0 or two_qubit_gate_seconds < 0.0:
        raise ValueError("Gate durations must be non-negative.")

    one_qubit_thermal = thermal_relaxation_error(
        t1_seconds,
        t2_seconds,
        one_qubit_gate_seconds,
    )
    one_qubit_error = depolarizing_error(depolarizing_1q, 1).compose(
        one_qubit_thermal
    )
    single_cx_thermal = thermal_relaxation_error(
        t1_seconds,
        t2_seconds,
        two_qubit_gate_seconds,
    )
    two_qubit_thermal = single_cx_thermal.tensor(single_cx_thermal)
    two_qubit_error = depolarizing_error(depolarizing_2q, 2).compose(
        two_qubit_thermal
    )

    noise_model = NoiseModel(basis_gates=list(HARDWARE_BASIS_GATES))
    noise_model.add_all_qubit_quantum_error(one_qubit_error, ["sx", "x"])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
    return noise_model


def run_gate_level_aer_distribution(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    shots: int = 8192,
    seed: int = 7,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample a hardware-basis circuit under an Aer gate-level noise model."""

    if shots <= 0:
        raise ValueError("shots must be positive.")
    measured = circuit.copy()
    measured.measure_all()
    backend = AerSimulator(noise_model=noise_model)
    compiled = transpile(
        measured,
        basis_gates=list(HARDWARE_BASIS_GATES),
        optimization_level=1,
        seed_transpiler=seed,
    )
    counts = backend.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result().get_counts()
    probabilities = np.zeros(1 << circuit.num_qubits, dtype=float)
    for bitstring, count in counts.items():
        probabilities[int(bitstring.replace(" ", ""), 2)] += count / shots
    operations = compiled.count_ops()
    metadata = {
        "transpiled_depth": int(compiled.depth()),
        "transpiled_cx": int(operations.get("cx", 0)),
        "transpiled_sx": int(operations.get("sx", 0)),
        "shots": int(shots),
    }
    return probabilities, metadata
