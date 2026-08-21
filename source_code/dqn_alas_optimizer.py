"""
dqn_alas_optimizer.py
---------------------
DQN-based automatic parameter tuning for the proposed ALAS-SCLF decoder.

Pure NumPy implementation -- NO TensorFlow / NO PyTorch required.
Works on Python 3.8+.

This optimizer exclusively targets decode_alas_sclf (our proposed solution).
The baseline decode_sclf and decode_ts_sclf functions are NEVER called or
modified here -- they are the paper comparisons.

Parameters tuned (3 total):
    D1    -- FLIPSKIP window size (controls skip region width)
    D2    -- FLIPSKIP count threshold (controls skip trigger)
    alpha -- SCL-RE early-stop threshold scale factor

Objective  : Minimize average SCL decoding attempts (complexity)
Constraint : FER <= FER_THRESHOLD (reliability must be preserved)

Reward function (exact TS-SCLF paper Equation 7):
    R(s) = T_prev - T_curr   if FER(s) <= (1+delta)*FER_ref
         = -1000             otherwise
where T_prev is the previous step's average attempts (or baseline on reset).
"""

import numpy as np
import random
import os

# ---------------------------------------------------------------------------
# Import the proposed ALAS-SCLF decoder from rigorous_polar.py (READ-ONLY)
# The baseline decode_sclf / decode_ts_sclf are NOT imported here at all.
# ---------------------------------------------------------------------------
from rigorous_polar import (
    PolarCode,
    decode_alas_sclf,       # <-- our proposed solution ONLY
    # NOTE: encode_segmented_crc from rigorous_polar is hardcoded for l_seg=24.
    # We use local encode_alas_crc (l_seg=16) instead for ALAS-SCLF.
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_PARAMS    = 3           # D1, D2, alpha (3-D state space)
SNR_DB        = 1.5         # Eb/N0 optimization point (dB) -- same as paper
FER_THRESHOLD = 0.02050     # TS-SCLF FER at 1.5 dB (from Table I); we must
                            # stay within delta=15% above: 0.02050 * 1.15
FER_DELTA     = 0.15        # 15% slack on FER constraint (Eq. 7 delta)
NUM_TRIALS    = 300         # Monte Carlo frames per evaluation
N, K          = 128, 56
L_SEG         = 16          # Asymmetric partition length (ALAS-SCLF proposal)

# CRC polynomials (same as rigorous_polar.py -- local copies to avoid import)
POLY_CRC3 = np.array([1, 0, 1, 1], dtype=int)       # x^3 + x + 1
POLY_CRC5 = np.array([1, 0, 0, 1, 0, 1], dtype=int) # x^5 + x^2 + 1

random.seed(42)
np.random.seed(42)

CODE = PolarCode(N=N, K=K, l_seg=L_SEG)   # l_seg=16: our asymmetric partition

# Pre-compute channel parameters for SNR_DB
R_CODE   = K / N
EBNO     = 10 ** (SNR_DB / 10.0)
SIGMA    = np.sqrt(1.0 / (2.0 * R_CODE * EBNO))

LOG_FILE = "dqn_alas_log.txt"


# ---------------------------------------------------------------------------
# Local CRC encoder matching the ALAS-SCLF asymmetric l_seg=16 partition
# (mirrors encode_segmented_crc_generic from alas_sclf_sweep.py)
# IMPORTANT: encode_segmented_crc in rigorous_polar.py is hardcoded for
# l_seg=24. We must NOT use it here -- we need l_seg=16 for ALAS-SCLF.
# ---------------------------------------------------------------------------
def _compute_crc(msg, poly):
    deg = len(poly) - 1
    padded = np.concatenate([msg, np.zeros(deg, dtype=int)])
    for i in range(len(msg)):
        if padded[i] == 1:
            padded[i : i + len(poly)] ^= poly
    return padded[-deg:]


def encode_alas_crc(info_payload):
    """
    Encodes K-8=48 info bits into K=56 bits using asymmetric segmented CRC
    with l_seg=16 (ALAS-SCLF partition):
      - CRC1 (3-bit) over first 16 bits  -> u_I[16:19]
      - CRC2 (5-bit) over first 51 bits  -> u_I[51:56]
    Matches exactly how alas_sclf_sweep.py encodes frames for ALAS-SCLF.
    """
    u_I = np.zeros(K, dtype=int)
    # Segment 1: first L_SEG info bits
    u_I[0 : L_SEG] = info_payload[0 : L_SEG]
    u_I[L_SEG : L_SEG + 3] = _compute_crc(u_I[0 : L_SEG], POLY_CRC3)
    # Segment 2: remaining info bits
    u_I[L_SEG + 3 : K - 5] = info_payload[L_SEG : K - 8]
    u_I[K - 5 : K] = _compute_crc(u_I[0 : K - 5], POLY_CRC5)
    return u_I


# ---------------------------------------------------------------------------
# Parameter Encoding / Decoding
# ---------------------------------------------------------------------------
# Internal state vector: s = [D1_scaled, D2_scaled, alpha_scaled]
# Actual parameters:
#   D1    = max(1, round(s[0] * 2))   -- step size 2 bits (range ~1..30)
#   D2    = max(1, round(s[1] * 1))   -- step size 1 count (range ~1..15)
#   alpha = max(0.1, s[2] / 4.0)      -- step size 0.25 (range ~0.1..3.0)

def state_to_params(s):
    D1    = int(max(1,   round(s[0] * 2)))
    D2    = int(max(1,   round(s[1] * 1)))
    alpha = float(max(0.1, s[2] / 4.0))
    return D1, D2, alpha


# ---------------------------------------------------------------------------
# Black-Box Evaluator
# ---------------------------------------------------------------------------
def evaluate_alas_sclf(D1, D2, alpha):
    """
    Monte Carlo evaluation of decode_alas_sclf with the given parameters.
    Returns (avg_attempts, fer, constraint_ok).

    IMPORTANT: Uses encode_alas_crc (l_seg=16) to match the ALAS-SCLF decoder
    which also uses CODE with l_seg=16.  This is the same encoder/decoder pair
    as alas_sclf_sweep.py lines 92-102.
    """
    fer_count = 0
    total_att = 0.0

    for _ in range(NUM_TRIALS):
        payload = np.random.randint(0, 2, K - 8)
        # Use asymmetric l_seg=16 encoder -- ALAS-SCLF specific
        u_I   = encode_alas_crc(payload)
        x     = CODE.encode(u_I)
        s_sig = 1 - 2 * x
        r     = s_sig + np.random.normal(0, SIGMA, N)
        y     = 2 * r / (SIGMA ** 2)

        # Call OUR proposed decoder only (never touches baseline decoders)
        dec, success, attempts = decode_alas_sclf(
            y, CODE, T=10, D1=D1, D2=D2, alpha=alpha, guard_band=8
        )

        if not success or not np.array_equal(dec, u_I):
            fer_count += 1
        total_att += attempts

    avg_att = total_att / NUM_TRIALS
    fer     = fer_count / NUM_TRIALS
    # Constraint: FER must not exceed (1+delta)*FER_ref (Eq. 7)
    ok      = fer <= FER_THRESHOLD * (1.0 + FER_DELTA)
    return avg_att, fer, ok


# ---------------------------------------------------------------------------
# Pure-NumPy DQN Network
# 2 hidden layers of 64 units, ReLU activations, linear output layer
# Xavier initialization for stable training
# ---------------------------------------------------------------------------
class NumpyDQN:
    """
    Small DQN Q-network implemented from scratch with NumPy.
    No external ML libraries required.
    Input:  state vector (NUM_PARAMS dimensions)
    Output: Q-values for all actions (NUM_PARAMS * 2 actions)
    """

    def __init__(self, input_dim, output_dim, lr=0.001):
        self.lr = lr

        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, (fan_in, fan_out))

        self.W1 = xavier(input_dim, 64);  self.b1 = np.zeros((1, 64))
        self.W2 = xavier(64, 64);         self.b2 = np.zeros((1, 64))
        self.W3 = xavier(64, output_dim); self.b3 = np.zeros((1, output_dim))

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.x0  = x
        self.z1  = x     @ self.W1 + self.b1;  self.a1 = self.relu(self.z1)
        self.z2  = self.a1 @ self.W2 + self.b2;  self.a2 = self.relu(self.z2)
        self.out = self.a2 @ self.W3 + self.b3
        return self.out

    def train(self, x, y_target):
        """One gradient-descent step on MSE loss."""
        y_pred = self.forward(x)
        diff   = y_pred - y_target           # shape: (1, output_dim)

        # Backprop through linear output -> ReLU2 -> ReLU1
        dL_dout = 2.0 * diff / diff.size
        dL_dW3  = self.a2.T @ dL_dout
        dL_db3  = dL_dout.sum(axis=0, keepdims=True)

        dL_da2  = dL_dout @ self.W3.T
        dL_dz2  = dL_da2 * self.relu_deriv(self.z2)
        dL_dW2  = self.a1.T @ dL_dz2
        dL_db2  = dL_dz2.sum(axis=0, keepdims=True)

        dL_da1  = dL_dz2 @ self.W2.T
        dL_dz1  = dL_da1 * self.relu_deriv(self.z1)
        dL_dW1  = self.x0.T @ dL_dz1
        dL_db1  = dL_dz1.sum(axis=0, keepdims=True)

        # Gradient descent weight update
        self.W3 -= self.lr * dL_dW3;  self.b3 -= self.lr * dL_db3
        self.W2 -= self.lr * dL_dW2;  self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1;  self.b1 -= self.lr * dL_db1


# ---------------------------------------------------------------------------
# RL Environment: ALAS-SCLF Parameter Tuning
# ---------------------------------------------------------------------------
class ALAS_SCLF_Env:
    """
    Reinforcement Learning environment wrapping decode_alas_sclf.

    State:  scaled parameter vector [D1_sc, D2_sc, alpha_sc]
    Action: one of 6 discrete actions:
            0: D1_sc += 1   (increase D1 by 2)
            1: D1_sc -= 1   (decrease D1 by 2)
            2: D2_sc += 1   (increase D2 by 1)
            3: D2_sc -= 1   (decrease D2 by 1)
            4: alpha_sc += 1 (increase alpha by 0.25)
            5: alpha_sc -= 1 (decrease alpha by 0.25)
    Reward (Exact TS-SCLF paper Eq. 7):
            R = T_prev - T_curr   if FER satisfied
              = -1000             otherwise
    """

    # Default initial operating point (D1=16, D2=3, alpha=1.0)
    INIT_STATE = np.array([8.0, 3.0, 4.0], dtype=float)  # scaled

    def __init__(self):
        self.current_state = np.array(self.INIT_STATE)
        self.prev_attempts  = None   # set on first evaluation
        self.best_state     = np.array(self.INIT_STATE)
        self.best_attempts  = 1e9
        self.action_space   = NUM_PARAMS * 2   # 6 actions

        # Evaluate baseline attempts at starting point (for T_prev on reset)
        print("  [ENV] Evaluating ALAS-SCLF baseline at starting point...")
        D1, D2, alpha = state_to_params(self.INIT_STATE)
        avg_att, fer, ok = evaluate_alas_sclf(D1, D2, alpha)
        self.baseline_attempts = avg_att
        print(f"  [ENV] Baseline: D1={D1}, D2={D2}, alpha={alpha:.3f} "
              f"=> AvgAtt={avg_att:.4f}, FER={fer:.5f}, OK={ok}")

    def reset(self):
        """Reset state to initial parameters. T_prev = baseline attempts."""
        self.current_state = np.array(self.INIT_STATE)
        self.prev_attempts  = self.baseline_attempts
        return np.array(self.current_state)

    def step(self, action):
        """
        Apply action, evaluate ALAS-SCLF, compute reward per Eq. 7, log result.
        """
        # Apply discrete action to scaled state vector
        param_idx = action // 2
        delta     = +1.0 if (action % 2 == 0) else -1.0
        self.current_state[param_idx] += delta

        # Clip to safe bounds (prevent degenerate parameter values)
        self.current_state[0] = np.clip(self.current_state[0],  1.0, 20.0)  # D1_sc
        self.current_state[1] = np.clip(self.current_state[1],  1.0, 15.0)  # D2_sc
        self.current_state[2] = np.clip(self.current_state[2],  1.0, 12.0)  # alpha_sc

        D1, D2, alpha = state_to_params(self.current_state)
        avg_att, fer, ok = evaluate_alas_sclf(D1, D2, alpha)

        # -------------------------------------------------------
        # Exact TS-SCLF paper Eq. 7 reward formulation
        # -------------------------------------------------------
        if ok:
            reward = self.prev_attempts - avg_att   # positive if we improved
            self.prev_attempts = avg_att            # update for next step
        else:
            reward = -1000.0                        # hard penalty for FER violation

        # Track global best feasible solution
        if ok and avg_att < self.best_attempts:
            self.best_attempts = avg_att
            self.best_state    = np.array(self.current_state)

        # Log to file in same format as dqn_lra_log.txt
        with open(LOG_FILE, 'a') as f:
            f.write(f"{self.current_state.tolist()} "
                    f"{avg_att:.5f} "
                    f"{ok}\n")

        return (
            np.array(self.current_state),
            reward,
            False,
            {"avg_att": avg_att, "fer": fer, "constraint_ok": ok},
        )


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    DQN Agent with epsilon-greedy exploration and experience replay.
    Architecture matches the existing dqn_lra_optimizer.py agent.
    """

    def __init__(self):
        self.memory        = []
        self.gamma         = 0.99          # discount factor
        self.epsilon       = 1.0           # initial exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min   = 0.01
        self.model = NumpyDQN(
            input_dim  = NUM_PARAMS,
            output_dim = NUM_PARAMS * 2,   # 6 Q-values (one per action)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(NUM_PARAMS * 2)
        q_vals = self.model.forward(state.reshape(1, -1))
        return int(np.argmax(q_vals[0]))

    def replay(self, batch_size=32):
        """Experience replay with Bellman target update."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            s  = state.reshape(1, -1)
            ns = next_state.reshape(1, -1)

            target_f         = self.model.forward(s).copy()
            bellman_target   = reward
            if not done:
                bellman_target += self.gamma * np.max(self.model.forward(ns))
            target_f[0][action] = bellman_target
            self.model.train(s, target_f)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  ALAS-SCLF DQN Parameter Optimizer  (pure NumPy, no ML libs)")
    print(f"  Proposed decoder: decode_alas_sclf  |  SNR = {SNR_DB} dB")
    print(f"  Parameters: D1, D2, alpha")
    print(f"  Reward: Eq.7 from TS-SCLF paper (T_prev - T_curr or -1000)")
    print(f"  FER Constraint: <= {FER_THRESHOLD * (1+FER_DELTA):.5f}")
    print(f"  Trials per evaluation: {NUM_TRIALS}")
    print(f"  Note: decode_sclf / decode_ts_sclf are baselines -- NOT modified")
    print("=" * 70)

    # Initialize log file with header
    with open(LOG_FILE, 'w') as f:
        f.write("state avg_attempts constraint_ok\n")

    env   = ALAS_SCLF_Env()
    agent = DQNAgent()

    total_episodes = 50   # 50 episodes to keep runtime manageable
    steps_per_ep   = 30

    for episode in range(total_episodes):
        state = env.reset()
        ep_reward = 0.0
        print(f"\n--- Episode {episode + 1}/{total_episodes} ---")

        for t in range(steps_per_ep):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            D1, D2, alpha = state_to_params(next_state)
            print(
                f"  t={t+1:2d}  D1={D1:3d}  D2={D2}  alpha={alpha:.3f}"
                f"  AvgAtt={info['avg_att']:.4f}  FER={info['fer']:.5f}"
                f"  R={reward:+.3f}  {'OK' if info['constraint_ok'] else 'FAIL'}"
            )

            agent.remember(state, action, reward, next_state, done)
            state     = next_state
            ep_reward += reward

            if done:
                break

        agent.replay(batch_size=32)

        # Epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        print(f"  Episode total reward: {ep_reward:.3f}  "
              f"  epsilon={agent.epsilon:.4f}")

    # Print final result
    best = env.best_state
    D1_best, D2_best, alpha_best = state_to_params(best)

    print("\n" + "=" * 70)
    print("  ALAS-SCLF DQN OPTIMIZATION COMPLETE")
    print(f"  Best D1    = {D1_best}")
    print(f"  Best D2    = {D2_best}")
    print(f"  Best alpha = {alpha_best:.4f}")
    print(f"  Best Avg Decoding Attempts = {env.best_attempts:.5f}")
    print(f"  Log written to: {LOG_FILE}")
    print("=" * 70)
