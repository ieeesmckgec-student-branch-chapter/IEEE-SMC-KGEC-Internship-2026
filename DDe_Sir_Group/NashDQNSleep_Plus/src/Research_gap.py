from builtins import set

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import warnings

from pathlib import Path
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

class IIoTEnvironment:
    """
    IIoT sensor network environment implementing both:
        - Original NashDQNSleep  (linear AoI,    Eq. 15  of [1])
        - NashDQNSleep+          (non-linear AoI, Eq. 15' proposed)

    Parameters mirror [1]: N nodes, small-world topology,
    reward weights w1-w5 as per Section 5.1 of [1].
    """

    def __init__(self, N=10, lam=0.0, mode='linear', p_loss=0.05,
                 w1=0.4, w2=0.4, w3=0.1, w4=0.1,w5=0.1,
                 delta_th=5.0, B_max=100.0, E_active=1.0, E_sleep=0.05):
        """
        Args:
            N       : Number of sensor nodes
            lam     : λ — non-linearity growth rate (0.0 = linear/original)
            mode    : 'linear' (original) or 'nonlinear' (NashDQNSleep+)
            p_loss  : Packet loss probability
            w1-w5   : Reward weights [1, Eq. 17]
            delta_th: AoI threshold Δ_th
            B_max   : Maximum battery level
            E_active: Energy consumed when active
            E_sleep : Energy consumed when sleeping
        """
        self.N        = N
        self.lam      = lam
        self.mode     = mode
        self.p_loss   = p_loss
        self.w1       = w1
        self.w2       = w2
        self.w3       = w3
        self.w4       = w4
        self.w5       = w5
        self.delta_th = delta_th
        self.B_max    = B_max
        self.E_active = E_active
        self.E_sleep  = E_sleep

        # Build neighbour adjacency (ring + random cross-links = small-world)
        self.neighbours = self._build_topology()
        self.reset()

    def _build_topology(self):

        """
        # Watts-Strogatz small-world topology as used in the
        #     original NashDQNSleep simulation.
        
        #     N = number of nodes
        #     k = 4 nearest neighbours
        #     p = 0.3 rewiring probability
        """
        import networkx as nx

        graph = nx.watts_strogatz_graph(
            n=self.N,
            k=4,
            p=0.3,
            seed=SEED
        )

        return {
                node: list(graph.neighbors(node))
                for node in graph.nodes()
            }

    def reset(self):
        """Initialise state as per [1, Section 7, p.14]."""
        self.B     = np.random.normal(self.B_max, 5, self.N).clip(30, self.B_max)
        self.Delta = np.ones(self.N)          # AoI initialised to 1
        self.s     = np.ones(self.N, dtype=int)  # all awake
        self.k     = 0
        return self._get_obs()

    # ── AoI update equations ──────────────────────────────────────────────────

    def _aoi_linear(self, n, success):
        """
        Original Eq. 15 of [1]:
            Δn(k+1) = 1              if active and successful
            Δn(k+1) = Δn(k) + 1     otherwise
        """
        if success:
            return 1.0
        return self.Delta[n] + 1.0

    def _aoi_nonlinear(self, n, success):
        """
        Proposed Eq. 15' (NashDQNSleep+):
            Δn(k+1) = 1                                    if active and successful
            Δn(k+1) = Δn(k) + exp(λ·(Δn(k) - 1))         otherwise

        When λ→0, exp(λ·(Δ−1)) → 1, recovering the linear model.
        """
        if success:
            return 1.0
        increment = np.exp(self.lam * (self.Delta[n] - 1.0))
        return self.Delta[n] + increment

    def _update_aoi(self, n, success):
        if self.mode == 'linear':
            return self._aoi_linear(n, success)
        return self._aoi_nonlinear(n, success)

    # ── Non-linear AoI cost φ(Δ) ─────────────────────────────────────────────

    def phi(self, delta):
        """
        Convex AoI cost function φ(Δ):
            φ(Δ) = (exp(λΔ) − 1) / λ    for λ > 0
            φ(Δ) = Δ                      for λ = 0 (linear case)
        """
        if self.lam == 0 or self.mode == 'linear':
            return delta
        return (np.exp(self.lam * delta) - 1.0) / self.lam

    # ── Reward function ───────────────────────────────────────────────────────

    def _reward(self, n, new_delta, success):
        """
        Original   [1, Eq. 17]:  R = w1·B − w2·Δ − w3·max(0,Δ−Δth) − w4·collision + w5·success
        NashDQNSleep+ Eq. 17' :  R = w1·B − w2·φ(Δ) − w3·max(0,φ(Δ)−Δth) − w4·collision + w5·success
        """
        # Active neighbour count → collision term
        N_active = sum(1 for m in self.neighbours[n] if self.s[m] == 1)
        collision = max(0, (N_active - 1) ** 2) if N_active > 1 else 0

        phi_val = self.phi(new_delta)

        r = (self.w1 * (self.B[n] / self.B_max)
             - self.w2 * phi_val
             - self.w3 * max(0.0, phi_val - self.delta_th)
             - self.w4 * collision
             + self.w5 * float(success))
        return float(r)

    # ── Environment step ──────────────────────────────────────────────────────

    def step(self, actions):
        """
        Execute one time step for all N nodes simultaneously.
        actions: list/array of {0: sleep, 1: awake} for each node
        Returns: obs, rewards, done, info
        """
        rewards    = np.zeros(self.N)
        new_Delta  = np.zeros(self.N)
        successes  = np.zeros(self.N, dtype=bool)

        for n in range(self.N):
            # Mode switch [1, Eq. 11]
            u = int(actions[n])
            self.s[n] = self.s[n] + u - 2 * self.s[n] * u

            success = False
            if self.s[n] == 1:
                # Transmission attempt with packet loss
                if np.random.rand() > self.p_loss:
                    success = True
                # Energy consumed when active
                self.B[n] = max(0.0, self.B[n] - self.E_active)
            else:
                # Sleep energy drain
                self.B[n] = max(0.0, self.B[n] - self.E_sleep)

            successes[n]   = success
            new_Delta[n]   = self._update_aoi(n, success)
            rewards[n]     = self._reward(n, new_Delta[n], success)

        self.Delta = new_Delta
        self.k    += 1

        obs  = self._get_obs()
        done = bool(np.any(self.B <= 0)) or self.k >= 300
        info = {
            'avg_aoi'    : float(np.mean(self.Delta)),
            'avg_battery': float(np.mean(self.B)),
            'n_active'   : int(np.sum(self.s)),
            'n_success'  : int(np.sum(successes))
        }
        return obs, rewards, done, info

    def _get_obs(self):
        """State vector Sn(k) = [Bn(k), sn(k-1), Δn(k), neighbour_avg_aoi]
           per [1, Eq. 16, p.6]."""
        obs = []
        for n in range(self.N):
            nb_aoi = np.mean([self.Delta[m] for m in self.neighbours[n]]) if self.neighbours[n] else 0.0
            obs.append([
                self.B[n] / self.B_max,   # normalised battery
                float(self.s[n]),          # current mode
                self.Delta[n] / 20.0,      # normalised AoI
                nb_aoi / 20.0              # neighbour AoI context
            ])
        return np.array(obs, dtype=np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # SECTION 2: INDEPENDENT DQN AGENT
#
# This implementation uses an online Q-network and a target Q-network
# with experience replay. It is a simplified DQN implementation used
# to study the proposed non-linear AoI modification.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QNetwork(nn.Module):
    """
    Two-hidden-layer DQN as per [1, Section 7.2]:
    2 layers × 64 neurons = ~10k weights, feasible on ARM Cortex-M4.
    """
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class NashDQNAgent:
    """
        Independent DQN agent for each node.

    This is a simplified DQN implementation used to evaluate the
    non-linear AoI extension proposed in this work. It is not a
    full reproduction of the Actor-Critic implementation described
    in the original NashDQNSleep paper.

        Each node maintains:
            - Q-network  θn  (online)
            - Q-network  θn⁻ (target)
            - Replay buffer D
    """
    def __init__(self, state_dim=4, action_dim=2,
                 lr=0.001, gamma=0.95, epsilon=1.0,
                 eps_decay=0.995, eps_min=0.01,
                 buffer_size=10000, batch_size=64):
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.eps_decay  = eps_decay
        self.eps_min    = eps_min
        self.batch_size = batch_size

        self.q_net      = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer     = deque(maxlen=buffer_size)
        self.loss_fn    = nn.MSELoss()

    def act(self, state):
        """ε-greedy action selection [1, Algorithm 1, line 23-27]."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self.q_net(s).argmax().item())

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def train_step(self):
        """DQN update [1, Eq. 27/53, Algorithm 1, lines 42-45]."""
        if len(self.buffer) < self.batch_size:
            return None

        batch  = random.sample(self.buffer, self.batch_size)
        S, A, R, S2, D = zip(*batch)

        S   = torch.FloatTensor(np.array(S))
        A   = torch.LongTensor(A).unsqueeze(1)
        R   = torch.FloatTensor(R)
        S2  = torch.FloatTensor(np.array(S2))
        D   = torch.FloatTensor(D)

        # Current Q values
        Q_curr = self.q_net(S).gather(1, A).squeeze()

        # Target Q values (Bellman)
        with torch.no_grad():
            Q_next  = self.target_net(S2).max(1)[0]
            Q_target = R + self.gamma * Q_next * (1 - D)

        loss = self.loss_fn(Q_curr, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self, tau=0.005):
        """Soft target update [1, nomenclature: τ = 0.005]."""
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(mode='linear', lam=0.0, n_episodes=30, N=10, label=''):
    """
    Run full NashDQNSleep training for given mode and λ.
    Returns episode-level metrics for plotting.
    """
    env    = IIoTEnvironment(N=N, lam=lam, mode=mode)
    agents = [NashDQNAgent() for _ in range(N)]

    ep_rewards   = []
    ep_aoi       = []
    ep_battery   = []
    ep_success   = []

    print(f"  Running [{label}] lam={lam:.3f}, mode={mode} ...")

    for ep in range(n_episodes):
        obs        = env.reset()
        total_r    = np.zeros(N)
        step_aoi   = []
        step_bat   = []
        step_succ  = []
        done       = False

        while not done:
            actions = [agents[n].act(obs[n]) for n in range(N)]
            obs2, rewards, done, info = env.step(actions)

            for n in range(N):
                agents[n].store(obs[n], actions[n], rewards[n], obs2[n], float(done))
                agents[n].train_step()
                agents[n].update_target()

            total_r   += rewards
            step_aoi.append(info['avg_aoi'])
            step_bat.append(info['avg_battery'])
            step_succ.append(info['n_success'] / N)
            obs = obs2

        for n in range(N):
            agents[n].decay_epsilon()

        ep_rewards.append(float(np.mean(total_r)))
        ep_aoi.append(float(np.mean(step_aoi)))
        ep_battery.append(float(np.mean(step_bat)))
        ep_success.append(float(np.mean(step_succ)))

    return {
        'rewards' : ep_rewards,
        'aoi'     : ep_aoi,
        'battery' : ep_battery,
        'success' : ep_success,
        'label'   : label
    }


def smooth(data, w=5):
    """Simple moving average for cleaner plots."""
    return np.convolve(data, np.ones(w)/w, mode='valid')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: ANALYTICAL CURVES (no training needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def aoi_growth_curve(steps, lam):
    """
    Simulate AoI growth analytically from Δ(0)=1
    under no transmissions — pure growth comparison.
    """
    delta = 1.0
    curve = [delta]
    for _ in range(steps - 1):
        if lam == 0:
            delta = delta + 1.0
        else:
            delta = delta + np.exp(lam * (delta - 1.0))
        curve.append(min(delta, 200))  # cap for display
    return np.array(curve)


def phi_curve(deltas, lam):
    """φ(Δ) = (exp(λΔ)−1)/λ  for λ>0, else Δ."""
    if lam == 0:
        return deltas
    return (np.exp(lam * deltas) - 1.0) / lam


def nash_reward_vs_lambda(lambdas, delta_fixed=3.0, B=0.7, w1=0.4, w2=0.4):
    """
    Show how expected reward changes with λ for a fixed AoI state δ.
    Demonstrates λ's effect on Nash equilibrium incentive to wake up.
    """
    rewards_sleep  = []
    rewards_active = []
    for lam in lambdas:
        phi = (np.exp(lam * delta_fixed) - 1.0) / lam if lam > 0 else delta_fixed
        # Sleep: no AoI reset, battery barely drained
        r_sleep  = w1 * (B - 0.05/100) - w2 * phi
        # Active: AoI resets to 1 with p=0.95
        phi_reset = (np.exp(lam * 1.0) - 1.0) / lam if lam > 0 else 1.0
        r_active = w1 * (B - 1.0/100) - w2 * (0.95 * phi_reset + 0.05 * phi)
        rewards_sleep.append(r_sleep)
        rewards_active.append(r_active)
    return np.array(rewards_sleep), np.array(rewards_active)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: MAIN — RUN ALL EXPERIMENTS + PLOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':

    N_EP = 30   # episodes per experiment

    print("=" * 30)
    print("NashDQNSleep+ Engineering Simulation")
    print("Gap 1: Non-Linear AoI Model")
    print("=" * 30)

    # ── Experiment A: Original NashDQNSleep (λ=0, linear) ────────────────────
    print("\n[Experiment A] Original NashDQNSleep — Linear AoI (λ=0)")
    res_linear = run_experiment(
        mode='linear', lam=0.0, n_episodes=N_EP,
        label='NashDQNSleep (Original, λ=0)'
    )

    # ── Experiment B: NashDQNSleep+ low λ ────────────────────────────────────
    print("\n[Experiment B] NashDQNSleep+ — Non-Linear AoI (λ=0.05)")
    res_nl_low = run_experiment(
        mode='nonlinear', lam=0.05, n_episodes=N_EP,
        label='NashDQNSleep+ (λ=0.05)'
    )

    # ── Experiment C: NashDQNSleep+ medium λ ─────────────────────────────────
    print("\n[Experiment C] NashDQNSleep+ — Non-Linear AoI (λ=0.15)")
    res_nl_mid = run_experiment(
        mode='nonlinear', lam=0.15, n_episodes=N_EP,
        label='NashDQNSleep+ (λ=0.15)'
    )

    # ── Experiment D: NashDQNSleep+ high λ ───────────────────────────────────
    print("\n[Experiment D] NashDQNSleep+ — Non-Linear AoI (λ=0.30)")
    res_nl_high = run_experiment(
        mode='nonlinear', lam=0.30, n_episodes=N_EP,
        label='NashDQNSleep+ (λ=0.30)'
    )

    all_results = [res_linear, res_nl_low, res_nl_mid, res_nl_high]
    colours     = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    styles      = ['-', '--', '-.', ':']
    W = 5  # smoothing window

    # ── FIGURE 1: AoI Growth Curves (Analytical) ─────────────────────────────
    print("\nGenerating Figure 1: AoI Growth Curves ...")
    steps   = np.arange(1, 51)
    lambdas_plot = [0.0, 0.05, 0.15, 0.30]
    labs    = ['Linear (λ=0, Original)', 'λ=0.05', 'λ=0.15', 'λ=0.30']
    cols    = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig1.suptitle('Figure 1: AoI Growth Comparison — Linear vs Non-Linear Models\n'
                  '(Eq. 15 of [1] vs Proposed Eq. 15\' of NashDQNSleep+)',
                  fontsize=12, fontweight='bold')

    # Left: raw AoI growth
    ax = axes[0]
    for lam, lab, col, ls in zip(lambdas_plot, labs, cols, styles):
        curve = aoi_growth_curve(len(steps)+1, lam)[1:]
        ax.plot(steps, curve, label=lab, color=col, linestyle=ls, linewidth=2)
    ax.set_xlabel('Time Steps (k)', fontsize=11)
    ax.set_ylabel('Age of Information Δn(k)', fontsize=11)
    ax.set_title('AoI Growth Without Transmission\n(No successful update)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 80)

    # Right: φ(Δ) penalty functions
    ax = axes[1]
    deltas = np.linspace(0, 10, 200)
    for lam, lab, col, ls in zip(lambdas_plot, labs, cols, styles):
        phi = phi_curve(deltas, lam)
        ax.plot(deltas, phi, label=lab, color=col, linestyle=ls, linewidth=2)
    ax.set_xlabel('AoI Value Δ', fontsize=11)
    ax.set_ylabel('AoI Cost φ(Δ)', fontsize=11)
    ax.set_title('Non-Linear AoI Cost Function φ(Δ)\n'
                 'φ(Δ) = (e^(λΔ)−1)/λ  [Proposed Eq. 17\']', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('/OUTPUT_DIR/fig1_aoi_growth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_aoi_growth.png")

    # ── FIGURE 2: λ Effect on Nash Equilibrium ────────────────────────────────
    print("Generating Figure 2: λ Effect on Nash Equilibrium ...")
    lam_range = np.linspace(0.001, 0.5, 200)
    r_sleep, r_active = nash_reward_vs_lambda(lam_range)
    delta_crossover = []  # find Δ at which agent switches decision

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Figure 2: Effect of λ on Nash Equilibrium and Reward Structure',
                  fontsize=12, fontweight='bold')

    # Left: reward vs lambda
    ax = axes[0]
    ax.plot(lam_range, r_active, color='#4CAF50', linewidth=2, label='Reward if Active')
    ax.plot(lam_range, r_sleep,  color='#F44336', linewidth=2, label='Reward if Sleep')
    ax.fill_between(lam_range, r_active, r_sleep,
                    where=(r_active > r_sleep), alpha=0.15, color='#4CAF50',
                    label='Wake preferred (Nash: u*=1)')
    ax.fill_between(lam_range, r_active, r_sleep,
                    where=(r_active <= r_sleep), alpha=0.15, color='#F44336',
                    label='Sleep preferred (Nash: u*=0)')
    ax.set_xlabel('Non-linearity Parameter λ', fontsize=11)
    ax.set_ylabel('Expected Reward', fontsize=11)
    ax.set_title('Wake/Sleep Reward Difference\nvs λ (fixed Δ=3.0, B=70%)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: reward difference (wake incentive)
    ax = axes[1]
    diff = r_active - r_sleep
    ax.plot(lam_range, diff, color='#9C27B0', linewidth=2.5)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.fill_between(lam_range, 0, diff, where=(diff > 0),
                    alpha=0.2, color='#4CAF50', label='Higher incentive to wake')
    ax.fill_between(lam_range, 0, diff, where=(diff <= 0),
                    alpha=0.2, color='#F44336', label='Higher incentive to sleep')
    ax.set_xlabel('Non-linearity Parameter λ', fontsize=11)
    ax.set_ylabel('ΔReward = R(active) − R(sleep)', fontsize=11)
    ax.set_title('Wake Incentive vs λ\n(Higher = stronger urgency to transmit)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: φ(Δ) at multiple AoI values showing urgency
    ax = axes[2]
    delta_vals = [1, 3, 5, 8, 10]
    for dv in delta_vals:
        phi_vals = [phi_curve(np.array([dv]), lam)[0] for lam in lam_range]
        ax.plot(lam_range, phi_vals, linewidth=2, label=f'Δ={dv}')
    ax.set_xlabel('Non-linearity Parameter λ', fontsize=11)
    ax.set_ylabel('AoI Penalty φ(Δ)', fontsize=11)
    ax.set_title('AoI Penalty Sensitivity to λ\nfor Different AoI Values', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig('/OUTPUT_DIR/fig2_lambda_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_lambda_effect.png")

    # ── FIGURE 3: Training Performance Comparison ─────────────────────────────
    print("Generating Figure 3: Training Performance Comparison ...")
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Figure 3: NashDQNSleep+ vs Original NashDQNSleep — Training Performance\n'
                  '(N=10 nodes, 30 episodes, Small-World Topology)',
                  fontsize=12, fontweight='bold')

    metrics = ['aoi', 'battery', 'rewards', 'success']
    titles  = [
        'Average AoI per Episode\n(Lower is Better)',
        'Average Battery Level per Episode\n(Higher is Better)',
        'Cumulative Reward per Episode\n(Higher is Better)',
        'Successful Transmission Rate per Episode\n(Higher is Better)'
    ]
    ylabels = ['Avg AoI (slots)', 'Avg Battery (%)', 'Avg Reward', 'Success Rate']

    for ax, metric, title, ylab in zip(axes.flat, metrics, titles, ylabels):
        for res, col, ls in zip(all_results, colours, styles):
            data = res[metric]
            sm   = smooth(data, W)
            x    = np.arange(len(sm)) + W//2
            ax.plot(x, sm, color=col, linestyle=ls, linewidth=2, label=res['label'])
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig('/OUTPUT_DIR/fig3_training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_training_comparison.png")

    # ── FIGURE 4: Final Performance Bar Chart ────────────────────────────────
    print("Generating Figure 4: Final Performance Bar Chart ...")
    last_n    = 10
    bar_names = ['Original\n(Linear, λ=0)', 'NashDQNSleep+\n(λ=0.05)',
                 'NashDQNSleep+\n(λ=0.15)', 'NashDQNSleep+\n(λ=0.30)']
    bar_aoi   = [np.mean(r['aoi'][-last_n:])     for r in all_results]
    bar_bat   = [np.mean(r['battery'][-last_n:])  for r in all_results]
    bar_succ  = [np.mean(r['success'][-last_n:])  for r in all_results]
    bar_rew   = [np.mean(r['rewards'][-last_n:])  for r in all_results]

    # Normalize for improvement % relative to original
    base_aoi  = bar_aoi[0]
    base_bat  = bar_bat[0]
    base_succ = bar_succ[0]

    fig4, axes = plt.subplots(1, 4, figsize=(16, 6))
    fig4.suptitle('Figure 4: Final Training Performance (Last 10 Episodes)\n'
                  'NashDQNSleep+ vs Original NashDQNSleep',
                  fontsize=12, fontweight='bold')

    datasets  = [bar_aoi, bar_bat, bar_succ, bar_rew]
    bar_titles = ['Avg AoI (↓ better)', 'Avg Battery % (↑ better)',
                  'Transmission Success (↑ better)', 'Avg Reward (↑ better)']

    for ax, data, title, col_list in zip(axes, datasets, bar_titles,
                                          [colours]*4):
        bars = ax.bar(bar_names, data, color=col_list, edgecolor='black',
                      linewidth=0.8, alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(bar_names)))
        ax.set_xticklabels(bar_names, fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 * max(data),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig4.savefig('/OUTPUT_DIR/fig4_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_bar_comparison.png")

    # ── FIGURE 5: Combined Summary Dashboard ─────────────────────────────────
    print("Generating Figure 5: Summary Dashboard ...")
    fig5 = plt.figure(figsize=(16, 10))
    fig5.suptitle('NashDQNSleep+: Engineering Solution Summary\n'
                  'Gap 1 — Non-Linear AoI Model (Proposed Eq. 15\' and Eq. 17\')',
                  fontsize=13, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.45, wspace=0.35)

    # Top-left: AoI growth
    ax = fig5.add_subplot(gs[0, 0])
    for lam, lab, col, ls in zip(lambdas_plot, labs, cols, styles):
        c = aoi_growth_curve(31, lam)[1:]
        ax.plot(np.arange(1, 31), c, label=lab, color=col, ls=ls, lw=2)
    ax.set_title('AoI Growth (No Tx)\nEq. 15 vs Eq. 15\'', fontsize=10, fontweight='bold')
    ax.set_xlabel('Steps'); ax.set_ylabel('Δn(k)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0, 70)

    # Top-middle: φ(Δ)
    ax = fig5.add_subplot(gs[0, 1])
    d  = np.linspace(0, 8, 200)
    for lam, lab, col, ls in zip(lambdas_plot, labs, cols, styles):
        ax.plot(d, phi_curve(d, lam), label=lab, color=col, ls=ls, lw=2)
    ax.set_title('AoI Cost φ(Δ)\nProposed Reward Term', fontsize=10, fontweight='bold')
    ax.set_xlabel('AoI Δ'); ax.set_ylabel('φ(Δ)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Top-right: Nash incentive vs lambda
    ax = fig5.add_subplot(gs[0, 2])
    ax.plot(lam_range, r_active - r_sleep, color='#9C27B0', lw=2.5)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.fill_between(lam_range, 0, r_active - r_sleep, where=(r_active > r_sleep),
                    alpha=0.2, color='green')
    ax.fill_between(lam_range, 0, r_active - r_sleep, where=(r_active <= r_sleep),
                    alpha=0.2, color='red')
    ax.set_title('Active vs Sleep Reward Difference λ\n(Δ=3, B=70%)', fontsize=10, fontweight='bold')
    ax.set_xlabel('λ'); ax.set_ylabel('ΔReward (Active−Sleep)'); ax.grid(True, alpha=0.3)

    # Bottom-left: AoI training
    ax = fig5.add_subplot(gs[1, 0])
    for res, col, ls in zip(all_results, colours, styles):
        sm = smooth(res['aoi'], W)
        ax.plot(np.arange(len(sm))+W//2, sm, color=col, ls=ls, lw=2, label=res['label'])
    ax.set_title('Avg AoI per Episode\n(Lower=Better)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('AoI'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bottom-middle: Battery training
    ax = fig5.add_subplot(gs[1, 1])
    for res, col, ls in zip(all_results, colours, styles):
        sm = smooth(res['battery'], W)
        ax.plot(np.arange(len(sm))+W//2, sm, color=col, ls=ls, lw=2, label=res['label'])
    ax.set_title('Avg Battery Level\n(Higher=Better)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('Battery %'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bottom-right: Success rate
    ax = fig5.add_subplot(gs[1, 2])
    for res, col, ls in zip(all_results, colours, styles):
        sm = smooth(res['success'], W)
        ax.plot(np.arange(len(sm))+W//2, sm, color=col, ls=ls, lw=2, label=res['label'])
    ax.set_title('Transmission Success Rate\n(Higher=Better)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('Success Rate'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig5.savefig('/OUTPUT_DIR/fig5_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_dashboard.png")

    # ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY — Final 10 Episodes")
    print("=" * 65)
    print(f"{'Method':<35} {'AoI':>6} {'Battery':>8} {'Success':>8} {'Reward':>8}")
    print("-" * 65)
    for r, aoi, bat, suc, rew in zip(all_results, bar_aoi, bar_bat, bar_succ, bar_rew):
        name = r['label'][:34]
        print(f"{name:<35} {aoi:>6.2f} {bat:>8.2f} {suc:>8.3f} {rew:>8.3f}")
    print("-" * 65)

    # Improvement % over baseline
    print("\nImprovement of NashDQNSleep+ over Original:")
    for r, aoi, bat, suc in zip(all_results[1:], bar_aoi[1:], bar_bat[1:], bar_succ[1:]):
        aoi_imp = (base_aoi - aoi) / base_aoi * 100
        bat_imp = (bat - base_bat) / base_bat * 100
        suc_imp = (suc - base_succ) / base_succ * 100
        print(f"  {r['label']:<32} | AoI ↓{aoi_imp:+.1f}%  Battery ↑{bat_imp:+.1f}%  Success ↑{suc_imp:+.1f}%")
    print("=" * 65)
    print("\nAll figures saved. Simulation complete.")