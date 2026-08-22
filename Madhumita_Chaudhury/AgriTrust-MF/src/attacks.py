"""A threat model in which detection can actually fail.

The original notebook drew attacker telemetry from ranges disjoint from the benign
ranges (attacker `U(0.82, 1.0)` against benign `Beta(1.2, 14)`, mean 0.079) and then
used the raw feature as the risk score. Classes never overlapped, so F1 was 1.0 by
construction on every seed.

Two changes make the problem real:

1. Benign nodes acquire legitimate reasons to look anomalous. Congestion depresses
   forwarding ratios, queueing inflates timing jitter, well-placed nodes genuinely
   advertise attractive routes, and dense neighbourhoods produce similar identity
   fingerprints. Benign confounders are what make WSN intrusion detection hard, and
   the original model had none.

2. Attackers get a `stealth` dial in [0, 1]. At 0 they behave exactly as in the
   original notebook. At 1 they are statistically indistinguishable from benign
   nodes. Intermediate values interpolate the signature and, for the dropping
   attacks, duty-cycle the malicious behaviour so a watchdog only sees it part of
   the time.

Detection is then evaluated with a detector *fitted on different networks* than the
one it is scored on, so it cannot be hand-tuned to the generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agritrust_core import ATTACK_TYPES, Config, node_natural_loss, sigmoid

FEATURE_COLUMNS = [
    "forwarding_deficit",
    "advert_anomaly",
    "timing_anomaly",
    "identity_similarity",
    "degree_ratio",
]


def _blend(rng, benign_value, attack_value, stealth):
    """Interpolate an attacker's observable towards the benign value."""
    return (1.0 - stealth) * attack_value + stealth * benign_value


def create_telemetry_realistic(wsn, cfg: Config, stealth: float = 0.0):
    """Telemetry with benign confounders and stealth-modulated attackers.

    Also returns, per node, the effective malicious drop probability implied by the
    stealth level, so that the packet simulator and the detector see a consistent
    adversary.
    """
    rng = wsn["rng"]
    graph = wsn["graph"]
    labels = wsn["attack_labels"]
    n = cfg.n_sensor_nodes

    degrees = np.array([graph.degree(node) for node in range(n)], dtype=float)
    degree_ratio = degrees / max(degrees.max(), 1.0)

    # Legitimate load heterogeneity: a minority of benign nodes are genuinely busy.
    congestion = rng.beta(1.5, 6.0, size=n)
    # Nodes close to the sink legitimately advertise short, attractive routes.
    positions = wsn["positions"]
    sink_distance = np.linalg.norm(positions - np.array([105.0, 50.0]), axis=1)
    proximity = 1.0 - sink_distance / max(sink_distance.max(), 1e-9)

    rows = []
    effective_drop = np.zeros(n)

    for node in range(n):
        natural_loss = node_natural_loss(graph, node)
        expected_forward = 1.0 - natural_loss

        # --- benign baselines, now with real confounders ----------------------
        benign_forward = np.clip(
            expected_forward * (1.0 - 0.55 * congestion[node]) + rng.normal(0, 0.05), 0.0, 1.0
        )
        benign_advert = np.clip(
            rng.beta(1.6, 9.0) + 0.45 * proximity[node] * rng.beta(2.0, 3.0), 0.0, 1.0
        )
        benign_timing = np.clip(
            rng.beta(1.6, 9.0) + 0.50 * congestion[node] * rng.beta(2.0, 3.0), 0.0, 1.0
        )
        benign_identity = np.clip(
            rng.beta(1.4, 10.0) + 0.40 * degree_ratio[node] * rng.beta(2.0, 4.0), 0.0, 1.0
        )

        forwarding_ratio = benign_forward
        advert_anomaly = benign_advert
        timing_anomaly = benign_timing
        identity_similarity = benign_identity

        label = labels[node]
        if label == "blackhole":
            # Duty cycling: a stealthy blackhole forwards during observation windows.
            duty = 1.0 - stealth
            observed = _blend(rng, benign_forward, rng.uniform(0.00, 0.07), stealth)
            forwarding_ratio = observed
            effective_drop[node] = 0.98 * duty + 0.02
        elif label == "selective_forwarding":
            duty = 1.0 - 0.7 * stealth
            forwarding_ratio = _blend(rng, benign_forward, rng.uniform(0.35, 0.65), stealth)
            effective_drop[node] = 0.45 * duty
        elif label == "sinkhole":
            advert_anomaly = _blend(rng, benign_advert, rng.uniform(0.82, 1.00), stealth)
            forwarding_ratio = min(
                forwarding_ratio, _blend(rng, benign_forward, rng.uniform(0.65, 0.88), stealth)
            )
            effective_drop[node] = 0.18 * (1.0 - 0.5 * stealth)
        elif label == "wormhole":
            timing_anomaly = _blend(rng, benign_timing, rng.uniform(0.82, 1.00), stealth)
            effective_drop[node] = 0.20 * (1.0 - 0.5 * stealth)
        elif label == "sybil":
            identity_similarity = _blend(
                rng, benign_identity, rng.uniform(0.82, 1.00), stealth
            )
            effective_drop[node] = 0.22 * (1.0 - 0.5 * stealth)

        forwarding_deficit = float(
            np.clip((expected_forward - forwarding_ratio) / max(expected_forward, 1e-6), 0, 1)
        )

        rows.append(
            {
                "node": node,
                "natural_loss": natural_loss,
                "expected_forward": expected_forward,
                "forwarding_ratio": forwarding_ratio,
                "forwarding_deficit": forwarding_deficit,
                "advert_anomaly": advert_anomaly,
                "timing_anomaly": timing_anomaly,
                "identity_similarity": identity_similarity,
                "degree_ratio": degree_ratio[node],
                "congestion_truth": congestion[node],
            }
        )

    telemetry = pd.DataFrame(rows).set_index("node")
    return telemetry, effective_drop


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def handcrafted_risk(telemetry):
    """The original notebook's estimator, applied to the new telemetry.

    Kept verbatim in spirit so we can show how a detector tuned against a separable
    generator behaves once the classes overlap.
    """
    excess_drop = telemetry["forwarding_deficit"].to_numpy()
    blackhole = sigmoid(14.0 * (excess_drop - 0.72))
    selective = np.exp(-0.5 * ((excess_drop - 0.43) / 0.17) ** 2) * (1.0 - blackhole)

    component = pd.DataFrame(
        {
            "blackhole": blackhole,
            "selective_forwarding": selective,
            "sinkhole": telemetry["advert_anomaly"].to_numpy(),
            "wormhole": telemetry["timing_anomaly"].to_numpy(),
            "sybil": telemetry["identity_similarity"].to_numpy(),
        },
        index=telemetry.index,
    )
    combined = 1.0 - np.prod(1.0 - np.clip(component.to_numpy(), 0, 1), axis=1)
    component["combined_risk"] = np.clip(combined, 0, 1)
    component["trust"] = 1.0 - component["combined_risk"]
    return component


class MulticlassLogisticDetector:
    """Small multinomial logistic regression trained by gradient descent.

    Deliberately dependency-free and deliberately weak: the point is an honest,
    fitted baseline, not the best possible intrusion detector. It is trained on
    networks generated from *different* seeds than the one it scores.
    """

    def __init__(self, classes=None, iterations=600, learning_rate=0.5, l2=1e-3):
        self.classes = list(classes) if classes else ["normal", *ATTACK_TYPES]
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights = None
        self.mean = None
        self.scale = None

    def _design(self, telemetry):
        features = telemetry[FEATURE_COLUMNS].to_numpy(dtype=float)
        if self.mean is None:
            self.mean = features.mean(axis=0)
            self.scale = features.std(axis=0) + 1e-9
        standardised = (features - self.mean) / self.scale
        return np.hstack([standardised, np.ones((len(standardised), 1))])

    def fit(self, telemetry, labels):
        design = self._design(telemetry)
        targets = np.zeros((len(design), len(self.classes)))
        index = {name: i for i, name in enumerate(self.classes)}
        for row, label in enumerate(labels):
            targets[row, index[label]] = 1.0

        self.weights = np.zeros((design.shape[1], len(self.classes)))
        for _ in range(self.iterations):
            logits = design @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = design.T @ (probabilities - targets) / len(design)
            gradient += self.l2 * self.weights
            self.weights -= self.learning_rate * gradient
        return self

    def predict_proba(self, telemetry):
        design = self._design(telemetry)
        logits = design @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def risk_frame(self, telemetry):
        """Same interface as `estimate_five_attack_risk` so the router is unchanged."""
        probabilities = self.predict_proba(telemetry)
        frame = pd.DataFrame(probabilities, columns=self.classes, index=telemetry.index)
        component = frame[ATTACK_TYPES].copy()
        component["combined_risk"] = np.clip(1.0 - frame["normal"].to_numpy(), 0, 1)
        component["trust"] = 1.0 - component["combined_risk"]
        return component


# ---------------------------------------------------------------------------
# Detection metrics that do not saturate
# ---------------------------------------------------------------------------


def roc_auc(scores, positives):
    """Rank-based AUC. Returns NaN if one class is absent."""
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    n_positive, n_negative = int(positives.sum()), int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties so identical scores do not inflate the estimate.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        summed = np.zeros(len(unique))
        np.add.at(summed, inverse, ranks)
        ranks = (summed / counts)[inverse]
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def average_precision(scores, positives):
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    if positives.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    hits = positives[order]
    cumulative_hits = np.cumsum(hits)
    precision = cumulative_hits / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / hits.sum())


def detection_report(risks, true_labels, threshold=0.5):
    """Per-attack AUC, average precision, and F1 at a fixed operating point."""
    component = risks[ATTACK_TYPES]
    max_score = component.max(axis=1)
    predicted = component.idxmax(axis=1).astype(object)
    predicted[max_score < threshold] = "normal"

    rows = []
    for attack in ATTACK_TYPES:
        truth = np.asarray(true_labels == attack)
        pred = np.asarray(predicted == attack)
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores = component[attack].to_numpy()
        rows.append(
            {
                "attack": attack,
                "auc": roc_auc(scores, truth),
                "average_precision": average_precision(scores, truth),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "min_positive_score": float(scores[truth].min()) if truth.any() else np.nan,
                "max_negative_score": float(scores[~truth].max()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["separation_gap"] = frame["min_positive_score"] - frame["max_negative_score"]

    malicious_truth = np.asarray(true_labels != "normal")
    binary_auc = roc_auc(risks["combined_risk"].to_numpy(), malicious_truth)
    return frame, binary_auc
