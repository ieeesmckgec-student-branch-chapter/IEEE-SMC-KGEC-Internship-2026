"""
dqn_lra_optimizer.py
--------------------
DQN-based automatic parameter tuning for the LRA-AST decoder.

Pure NumPy implementation -- NO TensorFlow / NO PyTorch required.
Works on any Python version including 3.14.

Parameters tuned (4 total):
    D1        -- FLIPSKIP window size
    D2        -- FLIPSKIP count threshold
    alpha     -- SCL-RE early stop threshold
    gamma_lra -- LRA-AST reliability scaling exponent  <-- NEW

Objective  : Minimize average SCL runs (decoding complexity)
Constraint : FER <= FER_THRESHOLD
"""

import numpy as np
import random
import os

# -----------------------------------------------------------------------
# Import your existing decoder from rigorous_polar.py
# -----------------------------------------------------------------------
from rigorous_polar import (
    PolarCode,
    encode_segmented_crc,
    decode_ts_sclf,
)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
Val           = 4          # D1, D2, alpha, gamma_lra
SNR_DB        = 1.5        # Eb/N0 point to optimize (dB)
FER_THRESHOLD = 0.108030   # FER must stay at or below this
NUM_TRIALS    = 300        # Frames per black-box call
N, K          = 128, 56

random.seed(0)
np.random.seed(0)

CODE = PolarCode(N=N, K=K, l_seg=24)   # build once

# -----------------------------------------------------------------------
# Black-box function  (replaces the .exe in original code)
# -----------------------------------------------------------------------
def black_box_function(Para):
    """
    Para[0]*2 -> D1,  Para[1]*1 -> D2,
    Para[2]/4 -> alpha,  Para[3]*1 -> gamma_lra
    Returns: (avg_complexity, constraint_ok)
    """
    D1        = max(1, int(round(Para[0] * 2)))
    D2        = max(1, int(round(Para[1] * 1)))
    alpha     = max(0.05, float(Para[2]) / 4.0)
    gamma_lra = max(0.1,  float(Para[3]) * 1.0)

    R     = K / N
    ebno  = 10 ** (SNR_DB / 10.0)
    sigma = np.sqrt(1.0 / (2.0 * R * ebno))

    fer_count = 0
    total_att = 0.0

    for _ in range(NUM_TRIALS):
        payload = np.random.randint(0, 2, K - 8)
        u_I     = encode_segmented_crc(payload)
        x       = CODE.encode(u_I)
        s       = 1 - 2 * x
        r       = s + np.random.normal(0, sigma, N)
        y       = 2 * r / (sigma ** 2)

        dec, success, attempts, _, _ = decode_ts_sclf(
            y, CODE, L=4, T=10,
            D1=D1, D2=D2, alpha=alpha,
            apply_constraint=True, guard_band=8,
            use_lra_ast=True, gamma_lra=gamma_lra,
        )

        if not success or not np.array_equal(dec, u_I):
            fer_count += 1
        total_att += attempts

    avg_complexity = total_att  / NUM_TRIALS
    fer            = fer_count  / NUM_TRIALS
    constraint_ok  = (fer <= FER_THRESHOLD)

    print(f"  D1={D1:3d} D2={D2} alpha={alpha:.3f} gamma={gamma_lra:.3f}"
          f"  =>  FER={fer:.5f}  AvgAtt={avg_complexity:.3f}"
          f"  {'[OK]' if constraint_ok else '[FAIL]'}")

    return avg_complexity, constraint_ok


# -----------------------------------------------------------------------
# Pure-NumPy DQN Network
# (2 hidden layers of 64 units, ReLU, linear output)
# -----------------------------------------------------------------------
class NumpyDQN:
    """Small DQN implemented from scratch with NumPy. No external ML lib."""

    def __init__(self, input_dim, output_dim, lr=0.001):
        self.lr = lr
        # Xavier initialisation
        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, (fan_in, fan_out))

        self.W1 = xavier(input_dim, 64);  self.b1 = np.zeros((1, 64))
        self.W2 = xavier(64, 64);         self.b2 = np.zeros((1, 64))
        self.W3 = xavier(64, output_dim); self.b3 = np.zeros((1, output_dim))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.x0  = x
        self.z1  = x   @ self.W1 + self.b1;  self.a1 = self.relu(self.z1)
        self.z2  = self.a1 @ self.W2 + self.b2;  self.a2 = self.relu(self.z2)
        self.out = self.a2 @ self.W3 + self.b3
        return self.out

    def train(self, x, y_target):
        """One gradient-descent step on MSE loss."""
        y_pred = self.forward(x)
        diff   = y_pred - y_target                       # (1, output)

        # Backprop
        dL_dout = 2 * diff / diff.size
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

        # Update weights (Adam-lite = plain SGD here for simplicity)
        self.W3 -= self.lr * dL_dW3; self.b3 -= self.lr * dL_db3
        self.W2 -= self.lr * dL_dW2; self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1; self.b1 -= self.lr * dL_db1


# -----------------------------------------------------------------------
# RL Environment
# -----------------------------------------------------------------------
class BlackBoxEnv:
    def __init__(self):
        # Initial scaled state: D1=16, D2=3, alpha=0.75, gamma_lra=1.2
        self.aa = np.array([16, 3, 0.75, 1.2], dtype=float) \
                / np.array([ 2, 1, 0.25, 1.0], dtype=float)

        self.current_state = np.array(self.aa)
        self.action_space  = Val * 2    # 8 actions
        self.best          = np.array(self.aa)
        self.bestObj       = 1e4

    def reset(self):
        self.current_state = np.array(self.aa)
        return self.current_state

    def step(self, action):
        param_index = action // 2
        change      = 1 if action % 2 == 0 else -1
        self.current_state[param_index] += change

        target_value, constraint_ok = black_box_function(self.current_state)

        if constraint_ok and target_value < self.bestObj:
            self.bestObj = target_value
            self.best    = np.array(self.current_state)

        with open("dqn_lra_log.txt", 'a') as f:
            f.write(f"{self.current_state.tolist()} {target_value:.5f} {constraint_ok}\n")

        reward = -target_value if constraint_ok else -1000.0
        return self.current_state, reward, False, {
            "target_value": target_value, "constraint_ok": constraint_ok
        }


# -----------------------------------------------------------------------
# DQN Agent  (pure NumPy)
# -----------------------------------------------------------------------
class DQNAgent:
    def __init__(self):
        self.memory        = []
        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min   = 0.01
        self.model         = NumpyDQN(input_dim=Val, output_dim=Val * 2)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(Val * 2)
        q = self.model.forward(state.reshape(1, -1))
        return int(np.argmax(q[0]))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            s  = state.reshape(1, -1)
            ns = next_state.reshape(1, -1)

            target_f          = self.model.forward(s).copy()
            target            = reward
            if not done:
                target += self.gamma * np.max(self.model.forward(ns))
            target_f[0][action] = target

            self.model.train(s, target_f)


# -----------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  DQN LRA-AST Parameter Optimizer  (pure NumPy, no ML libs)")
    print(f"  Tuning: D1, D2, alpha, gamma_lra  |  SNR = {SNR_DB} dB")
    print(f"  Constraint: FER <= {FER_THRESHOLD}  |  Trials/eval = {NUM_TRIALS}")
    print("=" * 65)

    with open("dqn_lra_log.txt", 'w') as f:
        f.write("state target_value constraint_ok\n")

    env   = BlackBoxEnv()
    agent = DQNAgent()

    for episode in range(1000):
        state = env.reset()
        print(f"\n--- Episode {episode + 1} ---")

        for t in range(30):
            action                      = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        agent.replay(32)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    best = env.best
    print("\n" + "=" * 65)
    print("  OPTIMIZATION COMPLETE")
    print(f"  Best D1        = {int(round(best[0] * 2))}")
    print(f"  Best D2        = {int(round(best[1] * 1))}")
    print(f"  Best alpha     = {best[2] / 4.0:.4f}")
    print(f"  Best gamma_lra = {best[3] * 1.0:.4f}")
    print(f"  Best Avg Complexity = {env.bestObj:.4f}")
    print("=" * 65)
