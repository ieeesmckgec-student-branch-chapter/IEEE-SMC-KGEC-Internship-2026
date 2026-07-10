import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# =====================================================================
# 1. Polar Code Construction & Encoding
# =====================================================================

def polarization_weight_construction(N, K):
    """
    Constructs polar code by selecting the K most reliable channels
    using the Polarization Weight (PW) construction method.
    """
    n = int(np.log2(N))
    beta = 2**0.25  # 1.1892
    weights = []
    for i in range(N):
        w = 0.0
        for j in range(n):
            bit = (i >> j) & 1
            w += bit * (beta**j)
        weights.append((w, i))
    
    # Sort by weight descending (most reliable first)
    weights.sort(key=lambda x: x[0], reverse=True)
    
    # The first K indices are information channels
    info_indices = [x[1] for x in weights[:K]]
    info_indices.sort()  # Keep in ascending order of index
    
    frozen_indices = set(range(N)) - set(info_indices)
    return info_indices, frozen_indices, [w for w, i in sorted(weights, key=lambda x: x[1])]

def get_generator_matrix(N):
    """
    Generates the polar generator matrix G = F^(tensor product n)
    """
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G

# =====================================================================
# 2. Cyclic Redundancy Check (CRC)
# =====================================================================

def compute_crc(msg, poly):
    """
    Computes CRC remainder in GF(2)
    """
    deg = len(poly) - 1
    padded = np.concatenate([msg, np.zeros(deg, dtype=int)])
    for i in range(len(msg)):
        if padded[i] == 1:
            padded[i : i + len(poly)] ^= poly
    return padded[-deg:]

def check_crc(seq, poly):
    """
    Checks if the sequence (message + CRC) has a valid CRC (remainder all 0)
    """
    deg = len(poly) - 1
    padded = np.copy(seq)
    for i in range(len(seq) - deg):
        if padded[i] == 1:
            padded[i : i + len(poly)] ^= poly
    return np.all(padded[-deg:] == 0)

# CRC Polynomials
POLY_CRC3 = np.array([1, 0, 1, 1], dtype=int)       # x^3 + x + 1
POLY_CRC5 = np.array([1, 0, 0, 1, 0, 1], dtype=int) # x^5 + x^2 + 1

def encode_segmented_crc(info_payload):
    """
    Encodes 48 information bits into 56 bits using segmented CRC:
    - CRC1 (3-bit) computed over first 24 bits -> placed at [24:27]
    - CRC2 (5-bit) computed over first 51 bits -> placed at [51:56]
    """
    u_I = np.zeros(56, dtype=int)
    # Segment 1: first 24 bits
    u_I[0:24] = info_payload[0:24]
    u_I[24:27] = compute_crc(u_I[0:24], POLY_CRC3)
    # Segment 2: next 24 bits
    u_I[27:51] = info_payload[24:48]
    u_I[51:56] = compute_crc(u_I[0:51], POLY_CRC5)
    return u_I

# =====================================================================
# 3. Polar Code Decoders & Path Management
# =====================================================================

class Path:
    def __init__(self, n, N):
        self.pm = 0.0
        self.llrs = np.zeros((n + 1, N))
        self.bits = np.zeros((n + 1, N), dtype=int)

def copy_path(path, n, N):
    new_path = Path(n, N)
    new_path.pm = path.pm
    new_path.llrs = np.copy(path.llrs)
    new_path.bits = np.copy(path.bits)
    return new_path

class PolarCodec:
    def __init__(self, N=128, K=56, use_lra_ast=False, gamma_lra=1.0):
        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        self.info_indices, self.frozen_indices, self.all_pw = polarization_weight_construction(N, K)
        self.G = get_generator_matrix(N)
        self.poly_crc3 = POLY_CRC3
        self.poly_crc5 = POLY_CRC5
        
        # Segment 1 check point: codeword index corresponding to the last bit of CRC1 (info index 26)
        self.l_seg1 = self.info_indices[26]
        
        # LRA-AST parameters
        self.use_lra_ast = use_lra_ast
        self.gamma_lra = gamma_lra
        
        # Precompute polarization weights for the information channels
        self.pw = np.array([self.all_pw[idx] for idx in self.info_indices])
        self.pw_mean = np.mean(self.pw)
        
        # Track complexity (decoded leaf bits)
        self.leaf_operations = 0

def update_paths_at_leaf(l, paths, codec, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer):
    """
    Leaf-level processing of paths: splits paths, updates metrics, sorts, and prunes.
    Also handles segmented CRC checks and early termination.
    """
    codec.leaf_operations += 1
    is_frozen = (l in codec.frozen_indices)
    L = codec.L
    
    new_paths = []
    for path in paths:
        llr = path.llrs[codec.n, l]
        hard_decision = 0 if llr >= 0 else 1
        
        if is_frozen:
            # Frozen bit: must be 0
            path.bits[codec.n, l] = 0
            path.pm += abs(llr) if hard_decision != 0 else 0.0
            new_paths.append(path)
        else:
            # Information bit
            # Check if this is the layer we need to flip in SCLF
            if is_flipping and l == flip_layer:
                # Force the opposite of the hard decision (no split)
                forced_decision = 1 - hard_decision
                path.bits[codec.n, l] = forced_decision
                path.pm += abs(llr)
                new_paths.append(path)
            else:
                # Split path into 0 and 1 branches
                # Branch 0
                path_0 = copy_path(path, codec.n, codec.N)
                path_0.bits[codec.n, l] = 0
                path_0.pm += abs(llr) if hard_decision != 0 else 0.0
                new_paths.append(path_0)
                
                # Branch 1
                path_1 = copy_path(path, codec.n, codec.N)
                path_1.bits[codec.n, l] = 1
                path_1.pm += abs(llr) if hard_decision != 1 else 0.0
                new_paths.append(path_1)
                
    # Sort new paths by path metric (ascending)
    new_paths.sort(key=lambda p: p.pm)
    
    # Store candidates before pruning for early termination range calculation
    candidate_pms = [p.pm for p in new_paths]
    
    # Apply early-stopping threshold (Eq. 6) in restarted SCL rounds (is_first_round=False)
    if not is_first_round and len(candidate_pms) > 0 and l in first_round_pms:
        pm_min_0 = first_round_pms[l][0]
        pm_max_0 = first_round_pms[l][-1]
        range_0 = pm_max_0 - pm_min_0
        
        current_alpha = alpha
        if codec.use_lra_ast and l in codec.info_indices:
            # Layer-Reliability-Aware SCL Termination (our proposed improvement)
            # Find the position of l in information indices to get its weight
            info_idx = codec.info_indices.index(l)
            w = codec.pw[info_idx]
            # Scale alpha: larger weight -> larger alpha (looser), smaller weight -> smaller alpha (stricter)
            current_alpha = alpha * ((w / codec.pw_mean) ** codec.gamma_lra)
            
        if candidate_pms[0] > pm_min_0 + current_alpha * range_0:
            # The best candidate is too poor compared to the first round; terminate early!
            paths[:] = []
            return
            
    # Prune to list size L
    paths[:] = new_paths[:L]
    
    # Record first-round path metrics and LLRs for later rounds
    if is_first_round and len(paths) > 0:
        first_round_pms[l] = candidate_pms
        # Record LLR of the best path at this leaf
        first_round_llrs[l] = paths[0].llrs[codec.n, l]
        
    # Segmented CRC check at l_seg1
    if l == codec.l_seg1:
        valid_paths = []
        for path in paths:
            # Extract decoded info bits in segment 1
            u_I_decoded = path.bits[codec.n, codec.info_indices[:27]]
            if check_crc(u_I_decoded, codec.poly_crc3):
                valid_paths.append(path)
        # Update paths
        if len(valid_paths) > 0:
            paths[:] = valid_paths
        else:
            # All paths failed CRC1; terminate early!
            paths[:] = []

def decode_node(d, j, paths, codec, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer):
    """
    Recursive Arikan Polar tree decoder.
    """
    if len(paths) == 0:
        return
        
    n = codec.n
    N = codec.N
    if d == n:
        update_paths_at_leaf(j, paths, codec, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer)
        return

    M = 2**(n - d - 1)
    L_sub = 2**(n - d)
    
    # Left child LLRs
    for path in paths:
        a = path.llrs[d, j * L_sub : j * L_sub + M]
        b = path.llrs[d, j * L_sub + M : (j + 1) * L_sub]
        # f operation: min-sum approximation
        path.llrs[d + 1, 2 * j * M : (2 * j + 1) * M] = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        
    # Decode left child
    decode_node(d + 1, 2 * j, paths, codec, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer)
    
    if len(paths) == 0:
        return
        
    # Right child LLRs
    for path in paths:
        a = path.llrs[d, j * L_sub : j * L_sub + M]
        b = path.llrs[d, j * L_sub + M : (j + 1) * L_sub]
        u_left = path.bits[d + 1, 2 * j * M : (2 * j + 1) * M]
        # g operation
        path.llrs[d + 1, (2 * j + 1) * M : (2 * j + 2) * M] = b + (1 - 2 * u_left) * a
        
    # Decode right child
    decode_node(d + 1, 2 * j + 1, paths, codec, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer)
    
    if len(paths) == 0:
        return
        
    # Combine bit decisions
    for path in paths:
        u_left = path.bits[d + 1, 2 * j * M : (2 * j + 1) * M]
        u_right = path.bits[d + 1, (2 * j + 1) * M : (2 * j + 2) * M]
        path.bits[d, j * L_sub : (j + 1) * L_sub] = np.concatenate([u_left ^ u_right, u_right])

# =====================================================================
# 4. Decoding Implementations (SCL, SCLF, TS-SCLF)
# =====================================================================

def decode_ca_scl(y, codec, L=4):
    """
    Standard CRC-Aided Successive Cancellation List (CA-SCL) decoder.
    """
    codec.L = L
    first_path = Path(codec.n, codec.N)
    first_path.llrs[0, :] = y
    
    paths = [first_path]
    first_round_pms = {}
    first_round_llrs = {}
    
    # Run recursive decoding (is_first_round=True, but we ignore early stopping and flipping)
    decode_node(0, 0, paths, codec, is_first_round=True, first_round_pms=first_round_pms, 
                first_round_llrs=first_round_llrs, alpha=999.0, is_flipping=False, flip_layer=-1)
    
    # Check full CRC for surviving paths
    for path in paths:
        u_I = path.bits[codec.n, codec.info_indices]
        if check_crc(u_I, codec.poly_crc5):
            return u_I, True, 1  # Success, 1 attempt
            
    # If no path passes CRC, return the best path's estimate
    if len(paths) > 0:
        return paths[0].bits[codec.n, codec.info_indices], False, 1
    return np.zeros(codec.K, dtype=int), False, 1

def run_flipskip(S, i, D1, D2):
    """
    Algorithm 1: Path-Flipping Skip Criterion (FLIPSKIP)
    """
    if i == 0:
        return 0
    kc = 0
    l = S[i]
    for k in range(i):
        # Count failed attempts in range [l - D1, l)
        if 0 < l - S[k] < D1:
            kc += 1
    if kc >= D2:
        return 1  # Skip this attempt
    return 0

def generate_flipping_set(first_round_llrs, codec, failed_seg1):
    """
    Generates the flipping set S, containing the information layers
    sorted in ascending order of absolute LLR (most uncertain first).
    """
    info_layers = list(codec.info_indices)
    if failed_seg1:
        # Constrain flipping candidates to layers before the first segmented CRC point
        info_layers = [l for l in info_layers if l <= codec.l_seg1]
        
    # Sort layers by absolute LLR (lower LLR means higher uncertainty)
    sorted_layers = sorted(info_layers, key=lambda l: abs(first_round_llrs.get(l, 999.0)))
    return sorted_layers

def decode_sclf(y, codec, L=4, T=10, D1=16, D2=3):
    """
    Standard SCLF Polar Decoder (as in the paper, baseline before TS-SCLF).
    - Uses the same CRC polynomial for final check.
    - Flipping set sorted by ascending |LLR| (same as TS-SCLF).
    - NO early termination between restarts (alpha=infinity).
    - NO segmented CRC pruning during restarts.
    - FLIPSKIP criterion (D1, D2) is applied (same as TS-SCLF).
    This must yield IDENTICAL FER to TS-SCLF, but higher complexity.
    """
    codec.L = L
    # Save segmented CRC state and disable it for SCLF
    orig_l_seg1 = codec.l_seg1
    codec.l_seg1 = codec.N  # Disable segmented CRC by pushing checkpoint past all bits

    # --- Round 0: Initial SCL Decoding ---
    first_path = Path(codec.n, codec.N)
    first_path.llrs[0, :] = y
    paths = [first_path]

    first_round_pms = {}
    first_round_llrs = {}

    # alpha=999 means no early termination in round 0 (standard SCL)
    decode_node(0, 0, paths, codec, is_first_round=True, first_round_pms=first_round_pms,
                first_round_llrs=first_round_llrs, alpha=999.0, is_flipping=False, flip_layer=-1)

    # Restore segmented CRC state
    codec.l_seg1 = orig_l_seg1

    # Check if initial SCL decoding succeeded
    for path in paths:
        u_I = path.bits[codec.n, codec.info_indices]
        if check_crc(u_I, codec.poly_crc5):
            return u_I, True, 1

    # Generate flipping set S (no segmented CRC constraint for SCLF)
    S = generate_flipping_set(first_round_llrs, codec, failed_seg1=False)

    attempts = 1
    # --- Flipping Loop ---
    for i in range(min(T, len(S))):
        attempts += 1

        # FLIPSKIP (same criterion as TS-SCLF)
        fout = run_flipskip(S, i, D1, D2)
        if fout == 1:
            continue

        flip_layer = S[i]
        restart_path = Path(codec.n, codec.N)
        restart_path.llrs[0, :] = y
        restart_paths = [restart_path]

        # Disable segmented CRC and early termination for each SCLF restart
        codec.l_seg1 = codec.N
        decode_node(0, 0, restart_paths, codec, is_first_round=False, first_round_pms=first_round_pms,
                    first_round_llrs=first_round_llrs, alpha=999.0, is_flipping=True, flip_layer=flip_layer)
        codec.l_seg1 = orig_l_seg1

        for path in restart_paths:
            u_I = path.bits[codec.n, codec.info_indices]
            if check_crc(u_I, codec.poly_crc5):
                return u_I, True, attempts

    return np.zeros(codec.K, dtype=int), False, attempts


def decode_ts_sclf(y, codec, L=4, T=10, D1=16, D2=3, alpha=1.0):
    """
    TS-SCLF Polar Decoder (reproduced) / LRA-AST TS-SCLF (if enabled in codec).
    """
    codec.L = L
    
    # --- Round 0: Initial SCL Decoding ---
    first_path = Path(codec.n, codec.N)
    first_path.llrs[0, :] = y
    paths = [first_path]
    
    first_round_pms = {}
    first_round_llrs = {}
    
    decode_node(0, 0, paths, codec, is_first_round=True, first_round_pms=first_round_pms, 
                first_round_llrs=first_round_llrs, alpha=alpha, is_flipping=False, flip_layer=-1)
    
    # Check if initial SCL decoding succeeded
    for path in paths:
        u_I = path.bits[codec.n, codec.info_indices]
        if check_crc(u_I, codec.poly_crc5):
            return u_I, True, 1  # Succeeded in 1 attempt!
            
    # SCL failed. Determine if it failed Segment 1 CRC
    failed_seg1 = (len(paths) == 0 or len(first_round_pms.get(codec.l_seg1, [])) == 0)
    
    # Generate flipping set S, sorted by ascending absolute LLR
    S = generate_flipping_set(first_round_llrs, codec, failed_seg1)
    
    attempts = 1
    # --- Flipping Loop (Attempts 1 to T) ---
    for i in range(min(T, len(S))):
        attempts += 1
        
        # Check FLIPSKIP
        fout = run_flipskip(S, i, D1, D2)
        if fout == 1:
            # Skip this attempt
            continue
            
        # Restart SCL with early termination and bit-flipping at S[i]
        flip_layer = S[i]
        restart_path = Path(codec.n, codec.N)
        restart_path.llrs[0, :] = y
        restart_paths = [restart_path]
        
        decode_node(0, 0, restart_paths, codec, is_first_round=False, first_round_pms=first_round_pms,
                    first_round_llrs=first_round_llrs, alpha=alpha, is_flipping=True, flip_layer=flip_layer)
                    
        # Check if any path passed final CRC
        for path in restart_paths:
            u_I = path.bits[codec.n, codec.info_indices]
            if check_crc(u_I, codec.poly_crc5):
                return u_I, True, attempts
                
    # If all attempts failed, return best available
    return np.zeros(codec.K, dtype=int), False, attempts

# =====================================================================
# 5. NumPy-Based Deep Q-Network (MLP with Backpropagation)
# =====================================================================

class NumpyMLP:
    """
    A lightweight, multi-layer perceptron (MLP) built entirely in NumPy.
    Used to approximate Q-values for DQN.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        # He initialization for ReLU
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, output_dim))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2) # ReLU
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3 # Linear output
        
    def train_step(self, X, Y, lr=0.005):
        # Forward pass
        q_values = self.forward(X)
        
        # Loss gradient (MSE loss)
        loss_grad = 2.0 * (q_values - Y) / X.shape[0] # shape (batch_size, output_dim)
        
        # Backprop through Layer 3
        dW3 = np.dot(self.a2.T, loss_grad)
        db3 = np.sum(loss_grad, axis=0, keepdims=True)
        
        # Backprop through Layer 2
        da2 = np.dot(loss_grad, self.W3.T)
        dz2 = da2 * (self.z2 > 0) # ReLU grad
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Backprop through Layer 1
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0) # ReLU grad
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Gradient clipping to prevent exploding gradients
        for dw in [dW1, dW2, dW3, db1, db2, db3]:
            np.clip(dw, -1.0, 1.0, out=dw)
            
        # Parameter updates
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

class DQNAgent:
    def __init__(self, state_dim=4, action_dim=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        self.learning_rate = 0.01
        self.model = NumpyMLP(state_dim, action_dim, hidden_dim=32)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
            
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.model.forward(state.reshape(1, -1))
        return np.argmax(q_values[0])
        
    def replay(self, batch_size=16):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_dim))
        targets = np.zeros((batch_size, self.action_dim))
        
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[idx] = state
            current_q = self.model.forward(state.reshape(1, -1))[0]
            
            target = reward
            if not done:
                next_q = self.model.forward(next_state.reshape(1, -1))[0]
                target += self.gamma * np.max(next_q)
                
            current_q[action] = target
            targets[idx] = current_q
            
        self.model.train_step(states, targets, lr=self.learning_rate)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =====================================================================
# 6. Environment and Simulation Runner
# =====================================================================

class TS_SCLF_Env:
    """
    Simulation Environment for RL Agent to optimize threshold parameters.
    State: [D1, D2, alpha_0, gamma]
    """
    def __init__(self, codec, target_snr=1.5, num_eval_trials=150):
        self.codec = codec
        self.target_snr = target_snr
        self.num_eval_trials = num_eval_trials
        
        # State: [D1, D2, alpha_0, gamma]
        # Initial values based on paper and intuition
        self.state = np.array([16.0, 3.0, 1.0, 1.0])
        
        # Save baseline performance
        self.baseline_fer = 0.0
        self.baseline_attempts = 0.0
        self.evaluate_baseline()
        
    def evaluate_baseline(self):
        """Evaluate performance of SCL without flipping to get baseline FER."""
        fer_count = 0
        total_attempts = 0
        for _ in range(self.num_eval_trials):
            # Generate random payload
            payload = np.random.randint(0, 2, 48)
            u_I = encode_segmented_crc(payload)
            u = np.zeros(self.codec.N, dtype=int)
            u[self.codec.info_indices] = u_I
            x = np.dot(u, self.codec.G) % 2
            
            # BPSK + AWGN
            s = 1 - 2 * x
            rate = 56.0 / 128.0
            ebno = 10**(self.target_snr / 10.0)
            sigma = np.sqrt(1.0 / (2.0 * rate * ebno))
            r = s + np.random.normal(0, sigma, self.codec.N)
            y = 2 * r / (sigma**2)
            
            # Decode SCL
            self.codec.leaf_operations = 0
            decoded_u_I, success, attempts = decode_ca_scl(y, self.codec, L=4)
            if not success or not np.array_equal(decoded_u_I, u_I):
                fer_count += 1
            total_attempts += self.codec.leaf_operations / self.codec.N
            
        self.baseline_fer = fer_count / self.num_eval_trials
        self.baseline_attempts = total_attempts / self.num_eval_trials
        print(f"[ENV] Baseline CA-SCL (L=4) FER at {self.target_snr}dB: {self.baseline_fer:.4f}")
        
    def step(self, action):
        """
        Actions:
        0, 1: D1 += 8, D1 -= 8
        2, 3: D2 += 1, D2 -= 1
        4, 5: alpha_0 += 0.25, alpha_0 -= 0.25
        6, 7: gamma += 0.2, gamma -= 0.2
        """
        # Save previous state
        prev_state = np.copy(self.state)
        
        # Apply action
        if action == 0: self.state[0] = min(32.0, self.state[0] + 8.0)
        elif action == 1: self.state[0] = max(4.0, self.state[0] - 8.0)
        elif action == 2: self.state[1] = min(6.0, self.state[1] + 1.0)
        elif action == 3: self.state[1] = max(1.0, self.state[1] - 1.0)
        elif action == 4: self.state[2] = min(3.0, self.state[2] + 0.25)
        elif action == 5: self.state[2] = max(0.25, self.state[2] - 0.25)
        elif action == 6: self.state[3] = min(2.0, self.state[3] + 0.2)
        elif action == 7: self.state[3] = max(0.0, self.state[3] - 0.2)
        
        # Configure codec
        D1 = int(self.state[0])
        D2 = int(self.state[1])
        alpha = self.state[2]
        self.codec.gamma_lra = self.state[3]
        
        # Evaluate current parameters on random trials
        fer_count = 0
        total_attempts = 0
        
        for _ in range(self.num_eval_trials):
            payload = np.random.randint(0, 2, 48)
            u_I = encode_segmented_crc(payload)
            u = np.zeros(self.codec.N, dtype=int)
            u[self.codec.info_indices] = u_I
            x = np.dot(u, self.codec.G) % 2
            
            s = 1 - 2 * x
            rate = 56.0 / 128.0
            ebno = 10**(self.target_snr / 10.0)
            sigma = np.sqrt(1.0 / (2.0 * rate * ebno))
            r = s + np.random.normal(0, sigma, self.codec.N)
            y = 2 * r / (sigma**2)
            
            # Decode using our proposed LRA-AST TS-SCLF (which uses codec's LRA-AST flags)
            self.codec.leaf_operations = 0
            decoded_u_I, success, attempts = decode_ts_sclf(y, self.codec, L=4, T=10, D1=D1, D2=D2, alpha=alpha)
            
            # Error if CRC check fails OR if the decoded payload doesn't match the original
            if not success or not np.array_equal(decoded_u_I, u_I):
                fer_count += 1
            total_attempts += self.codec.leaf_operations / self.codec.N
            
        current_fer = fer_count / self.num_eval_trials
        current_attempts = total_attempts / self.num_eval_trials
        
        # Reward function:
        # Lower average attempts is rewarded, but only if FER is close to or better than baseline!
        # If FER degrades too much, we apply a large penalty.
        fer_satisfied = (current_fer <= self.baseline_fer * 1.15) # Allow 15% tolerance
        
        if fer_satisfied:
            # Reward proportional to reduction in decoding attempts compared to baseline
            reward = (self.baseline_attempts - current_attempts) * 10.0
        else:
            reward = -200.0  # Large penalty for degrading error-correction performance
            
        return self.state, reward, False, {"fer": current_fer, "attempts": current_attempts}

# =====================================================================
# 7. Main Execution & Comparison
# =====================================================================

if __name__ == "__main__":
    print("=================================================================")
    print("      POLAR CODES SIMULATION AND RL THRESHOLD OPTIMIZATION")
    print("=================================================================")
    
    # 1. Initialize Codec
    # We construct a (128, 56) polar code (which carries 48 user info bits)
    N = 128
    K = 56
    codec = PolarCodec(N, K, use_lra_ast=True, gamma_lra=1.0)
    print(f"Polar code constructed: N={N}, K={K} (48-bit payload, 8-bit segmented CRC)")
    print(f"Segmentation point at codeword index: {codec.l_seg1}")
    
    # 2. Train RL Agent to optimize parameters
    print("\n--- Training NumPy-based RL Agent to Optimize Thresholds ---")
    env = TS_SCLF_Env(codec, target_snr=1.5, num_eval_trials=100)
    agent = DQNAgent(state_dim=4, action_dim=8)
    
    best_state = np.copy(env.state)
    best_reward = -999.0
    
    # We do a quick training loop (30 episodes, 10 steps each)
    # This is extremely fast in pure Python
    for ep in range(30):
        state = np.copy(env.state)
        total_reward = 0.0
        for step in range(10):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = np.copy(next_state)
            total_reward += reward
            
            if reward > best_reward and info["fer"] <= env.baseline_fer * 1.15:
                best_reward = reward
                best_state = np.copy(state)
                
        agent.replay(batch_size=8)
        print(f"Episode {ep+1}/30 - Epsilon: {agent.epsilon:.2f} - Last Reward: {total_reward:.2f} - Current State: {state}")
        
    print(f"\nOptimization Finished!")
    print(f"Best Learned State [D1, D2, alpha_0, gamma]: {best_state}")
    print(f"Best Learned Parameters: D1={int(best_state[0])}, D2={int(best_state[1])}, alpha_0={best_state[2]:.2f}, gamma={best_state[3]:.2f}")
    
    # Configure the codec with the learned best state
    learned_D1 = int(best_state[0])
    learned_D2 = int(best_state[1])
    learned_alpha = best_state[2]
    learned_gamma = best_state[3]
    
    # 3. Comprehensive Performance Evaluation
    # We evaluate and compare 4 configurations:
    # 1. Baseline CA-SCL (L=4)
    # 2. Standard SCLF (L=4, T=10, no early stop)
    # 3. Reproduced TS-SCLF (L=4, T=10, global alpha)
    # 4. Proposed LRA-AST TS-SCLF (L=4, T=10, layer-adaptive alpha)
    
    snr_db_list = [0.5, 1.0, 1.5, 2.0, 2.5]
    trials_per_snr = 800  # High enough to get reliable statistics quickly
    
    results = {
        "CA-SCL": {"fer": [], "attempts": []},
        "SCLF": {"fer": [], "attempts": []},
        "TS-SCLF": {"fer": [], "attempts": []},
        "LRA-AST": {"fer": [], "attempts": []}
    }
    
    print("\n--- Running Comparative Monte Carlo Simulations ---")
    for snr_db in snr_db_list:
        print(f"Simulating SNR = {snr_db:.1f} dB...")
        
        ebno = 10**(snr_db / 10.0)
        rate = 56.0 / 128.0
        sigma = np.sqrt(1.0 / (2.0 * rate * ebno))
        
        counts = {k: {"fer": 0, "attempts": 0.0} for k in results.keys()}
        
        for _ in range(trials_per_snr):
            # Generate payload and encode
            payload = np.random.randint(0, 2, 48)
            u_I = encode_segmented_crc(payload)
            u = np.zeros(N, dtype=int)
            u[codec.info_indices] = u_I
            x = np.dot(u, codec.G) % 2
            
            # AWGN Channel
            s = 1 - 2 * x
            r = s + np.random.normal(0, sigma, N)
            y = 2 * r / (sigma**2)
            
            # --- 1. CA-SCL ---
            codec.leaf_operations = 0
            dec_u_I, success, attempts = decode_ca_scl(y, codec, L=4)
            if not success or not np.array_equal(dec_u_I, u_I):
                counts["CA-SCL"]["fer"] += 1
            counts["CA-SCL"]["attempts"] += codec.leaf_operations / N
            
            # --- 2. Standard SCLF (same D1/D2/T, but NO early stop, NO segmented CRC) ---
            # SCLF and TS-SCLF must have IDENTICAL FER (see paper Fig. 3).
            # The only difference is TS-SCLF uses early termination + segmented CRC to reduce complexity.
            codec.use_lra_ast = False
            codec.leaf_operations = 0
            dec_u_I, success, attempts = decode_sclf(y, codec, L=4, T=10, D1=learned_D1, D2=learned_D2)
            if not success or not np.array_equal(dec_u_I, u_I):
                counts["SCLF"]["fer"] += 1
            counts["SCLF"]["attempts"] += codec.leaf_operations / N
            
            # --- 3. Reproduced TS-SCLF (Global alpha, segmented CRC, same D1/D2 as SCLF) ---
            # Must yield same FER as SCLF, but lower complexity (fewer restarts due to early termination).
            codec.use_lra_ast = False
            codec.leaf_operations = 0
            dec_u_I, success, attempts = decode_ts_sclf(y, codec, L=4, T=10, D1=learned_D1, D2=learned_D2, alpha=learned_alpha)
            if not success or not np.array_equal(dec_u_I, u_I):
                counts["TS-SCLF"]["fer"] += 1
            counts["TS-SCLF"]["attempts"] += codec.leaf_operations / N
            
            # --- 4. Proposed LRA-AST TS-SCLF (Layer-reliability-adaptive alpha) ---
            codec.use_lra_ast = True
            codec.gamma_lra = learned_gamma
            codec.leaf_operations = 0
            dec_u_I, success, attempts = decode_ts_sclf(y, codec, L=4, T=10, D1=learned_D1, D2=learned_D2, alpha=learned_alpha)
            if not success or not np.array_equal(dec_u_I, u_I):
                counts["LRA-AST"]["fer"] += 1
            counts["LRA-AST"]["attempts"] += codec.leaf_operations / N
            
        # Store results
        for k in results.keys():
            results[k]["fer"].append(counts[k]["fer"] / trials_per_snr)
            results[k]["attempts"].append(counts[k]["attempts"] / trials_per_snr)
            
    # Print beautiful ASCII summary table
    print("\n" + "="*85)
    print(f"{'SNR (dB)':<10}{'Scheme':<15}{'FER':<15}{'Normalized Complexity':<25}{'Complexity Reduction vs SCLF':<25}")
    print("="*85)
    for idx, snr_db in enumerate(snr_db_list):
        sclf_attempts = results["SCLF"]["attempts"][idx]
        for scheme in results.keys():
            fer = results[scheme]["fer"][idx]
            attempts = results[scheme]["attempts"][idx]
            reduction = ((sclf_attempts - attempts) / sclf_attempts * 100.0) if scheme != "CA-SCL" else 0.0
            reduction_str = f"{reduction:.1f}%" if scheme in ["TS-SCLF", "LRA-AST"] else "-"
            print(f"{snr_db:<10.1f}{scheme:<15}{fer:<15.5f}{attempts:<25.3f}{reduction_str:<25}")
        print("-"*85)
        
    # Generate and save the performance plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. FER Plot
    colors = {"CA-SCL": "#808080", "SCLF": "#FF4500", "TS-SCLF": "#1E90FF", "LRA-AST": "#32CD32"}
    markers = {"CA-SCL": "x", "SCLF": "o", "TS-SCLF": "s", "LRA-AST": "^"}
    linestyles = {"CA-SCL": "--", "SCLF": "-", "TS-SCLF": "-", "LRA-AST": "-"}
    
    for scheme in results.keys():
        ax1.semilogy(snr_db_list, results[scheme]["fer"], label=scheme, 
                     color=colors[scheme], marker=markers[scheme], linestyle=linestyles[scheme], linewidth=2)
    ax1.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax1.set_ylabel("Frame Error Rate (FER)", fontsize=12)
    ax1.set_title("Frame Error Rate (FER) Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # 2. Average Decoding Attempts Plot
    for scheme in ["SCLF", "TS-SCLF", "LRA-AST"]:
        ax2.plot(snr_db_list, results[scheme]["attempts"], label=scheme, 
                 color=colors[scheme], marker=markers[scheme], linestyle=linestyles[scheme], linewidth=2)
    ax2.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax2.set_ylabel("Normalized Complexity (Equivalent SCL Runs)", fontsize=12)
    ax2.set_title("Decoding Complexity (Normalized SCL Runs)", fontsize=14, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = "polar_comparison_results.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved comparative plot to: {os.path.abspath(plot_path)}")
    print("=================================================================")
