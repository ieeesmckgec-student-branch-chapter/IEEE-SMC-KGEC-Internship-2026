import numpy as np
import random

# =====================================================================
# 1. Polar Code Construction & Encoding (GF(2) Recursive Convention)
# =====================================================================

def polarization_weight_construction(N, K):
    """
    Selects the K most reliable channels using the Polarization Weight (PW) method.
    """
    n = int(np.log2(N))
    beta = 2**0.25  # 1.189207115
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
    info_indices.sort()  # Keep in ascending order
    
    frozen_indices = set(range(N)) - set(info_indices)
    return info_indices, frozen_indices

class PolarCode:
    def __init__(self, N=128, K=56, l_seg=24):
        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        self.info_indices, self.frozen_indices = polarization_weight_construction(N, K)
        self.l_seg = l_seg
        # Segment 1 check point: codeword index corresponding to the last bit of CRC1 (info index l_seg + 2)
        self.l_seg1 = self.info_indices[l_seg + 2]
        
        # Precompute polarization weights for LRA-AST
        beta = 2**0.25
        self.weights = {}
        for i in range(N):
            w = 0.0
            for j in range(self.n):
                bit = (i >> j) & 1
                w += bit * (beta**j)
            self.weights[i] = w
            
        self.info_weights = [self.weights[idx] for idx in self.info_indices]
        self.w_mean = np.mean(self.info_weights)
        
    def encode(self, u_I):
        """
        Recursive Polar Encoder.
        Computes x = u * G in O(N log N) operations using the Arikan recursive convention.
        """
        u = np.zeros(self.N, dtype=int)
        u[self.info_indices] = u_I
        
        x = np.copy(u)
        for stage in range(self.n):
            step = 1 << stage
            for i in range(0, self.N, 2 * step):
                for j in range(step):
                    x[i + j] = (x[i + j] ^ x[i + j + step])
        return x

# =====================================================================
# 2. Cyclic Redundancy Check (CRC) in GF(2)
# =====================================================================

def compute_crc(msg, poly):
    """
    Computes CRC remainder in GF(2) using polynomial division.
    """
    deg = len(poly) - 1
    padded = np.concatenate([msg, np.zeros(deg, dtype=int)])
    for i in range(len(msg)):
        if padded[i] == 1:
            padded[i : i + len(poly)] ^= poly
    return padded[-deg:]

def check_crc(seq, poly):
    """
    Verifies if the sequence (message + CRC) has a valid CRC (remainder all 0).
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
    - CRC1 (3-bit) over first 24 bits -> u_I[24:27]
    - CRC2 (5-bit) over first 51 bits -> u_I[51:56]
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
# 3. Recursive Successive Cancellation List (SCL) & SCLF Decoder
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

def update_paths_at_leaf(l, paths, code, L, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer, guard_band=12, use_lra_ast=False, gamma_lra=0.8):
    """
    Leaf-level processing of paths: splits paths, updates metrics, sorts, and prunes.
    Also handles SCL-RE early stopping, LLR-ranking recording, and Segmented CRC checks.
    """
    is_frozen = (l in code.frozen_indices)
    
    new_paths = []
    for path in paths:
        llr = path.llrs[code.n, l]
        hard_decision = 0 if llr >= 0 else 1
        
        if is_frozen:
            # Frozen bit: must be 0
            path.bits[code.n, l] = 0
            path.pm += abs(llr) if hard_decision != 0 else 0.0
            new_paths.append(path)
        else:
            # Information bit
            if is_flipping and l == flip_layer:
                # Force the opposite of the hard decision (no path split)
                forced_decision = 1 - hard_decision
                path.bits[code.n, l] = forced_decision
                path.pm += abs(llr)
                new_paths.append(path)
            else:
                # Split path into 0 and 1 branches
                # Branch 0
                path_0 = copy_path(path, code.n, code.N)
                path_0.bits[code.n, l] = 0
                path_0.pm += abs(llr) if hard_decision != 0 else 0.0
                new_paths.append(path_0)
                
                # Branch 1
                path_1 = copy_path(path, code.n, code.N)
                path_1.bits[code.n, l] = 1
                path_1.pm += abs(llr) if hard_decision != 1 else 0.0
                new_paths.append(path_1)
                
    # Sort new paths by path metric (ascending)
    new_paths.sort(key=lambda p: p.pm)
    
    candidate_pms = [p.pm for p in new_paths]
    
    # --- SCL-RE Early Termination (Restarted Runs only, Information Layers only) ---
    if not is_first_round and len(candidate_pms) > 0 and not is_frozen and first_round_pms is not None:
        if l in first_round_pms and is_flipping and l > flip_layer + guard_band:
            pm_min_0 = first_round_pms[l][0]
            pm_max_0 = first_round_pms[l][-1]
            range_0 = pm_max_0 - pm_min_0
            
            # Layer-Reliability-Aware Adaptive SCL Termination (LRA-AST)
            current_alpha = alpha
            if use_lra_ast and hasattr(code, 'weights') and hasattr(code, 'w_mean'):
                w = code.weights[l]
                current_alpha = alpha * ((w / code.w_mean) ** gamma_lra)
            
            # Equation 6 from the paper (with adaptive threshold alpha(l) if enabled)
            if candidate_pms[0] > pm_min_0 + current_alpha * range_0:
                paths[:] = []
                if hasattr(code, 'early_termination_count'):
                    code.early_termination_count += 1
                return
                
    # Prune to list size L
    paths[:] = new_paths[:L]
    
    # Record first-round metrics (Information Layers only)
    if is_first_round and len(paths) > 0 and not is_frozen:
        if first_round_llrs is not None:
            first_round_llrs[l] = paths[0].llrs[code.n, l]
        if first_round_pms is not None:
            first_round_pms[l] = candidate_pms
            
    # --- Segmented CRC Check (During both initial and restarted runs) ---
    if l == code.l_seg1 and len(paths) > 0:
        valid_paths = []
        l_seg = code.l_seg if hasattr(code, 'l_seg') else 24
        for path in paths:
            # Extract decoded info bits in segment 1 (indices 0 to l_seg + 2 of info indices)
            u_I_decoded = path.bits[code.n, code.info_indices[: l_seg + 3]]
            if check_crc(u_I_decoded, POLY_CRC3):
                valid_paths.append(path)
        if len(valid_paths) > 0:
            paths[:] = valid_paths
        else:
            # All paths failed Segment 1 CRC; terminate run immediately!
            paths[:] = []
            return

def decode_node(d, j, paths, code, L, is_first_round=True, first_round_pms=None, first_round_llrs=None, alpha=1.0, is_flipping=False, flip_layer=-1, guard_band=12, use_lra_ast=False, gamma_lra=0.8):
    """
    Recursive Arikan Polar tree decoder.
    """
    if len(paths) == 0:
        return
        
    n = code.n
    N = code.N
    if d == n:
        update_paths_at_leaf(j, paths, code, L, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer, guard_band, use_lra_ast, gamma_lra)
        return

    M = 2**(n - d - 1)
    L_sub = 2**(n - d)
    
    # Left child LLRs: f operation (min-sum approximation)
    for path in paths:
        a = path.llrs[d, j * L_sub : j * L_sub + M]
        b = path.llrs[d, j * L_sub + M : (j + 1) * L_sub]
        path.llrs[d + 1, 2 * j * M : (2 * j + 1) * M] = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        
    # Decode left child
    decode_node(d + 1, 2 * j, paths, code, L, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer, guard_band, use_lra_ast, gamma_lra)
    
    if len(paths) == 0:
        return
        
    # Right child LLRs: g operation
    for path in paths:
        a = path.llrs[d, j * L_sub : j * L_sub + M]
        b = path.llrs[d, j * L_sub + M : (j + 1) * L_sub]
        u_left = path.bits[d + 1, 2 * j * M : (2 * j + 1) * M]
        path.llrs[d + 1, (2 * j + 1) * M : (2 * j + 2) * M] = b + (1 - 2 * u_left) * a
        
    # Decode right child
    decode_node(d + 1, 2 * j + 1, paths, code, L, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer, guard_band, use_lra_ast, gamma_lra)
    
    if len(paths) == 0:
        return
        
    # Combine bit decisions: matches Arikan recursive encoder combining [u_left ^ u_right, u_right]
    for path in paths:
        u_left = path.bits[d + 1, 2 * j * M : (2 * j + 1) * M]
        u_right = path.bits[d + 1, (2 * j + 1) * M : (2 * j + 2) * M]
        path.bits[d, j * L_sub : (j + 1) * L_sub] = np.concatenate([u_left ^ u_right, u_right])

def decode_ca_scl(y, code, L=4):
    """
    CRC-Aided Successive Cancellation List (CA-SCL) decoder.
    """
    first_path = Path(code.n, code.N)
    first_path.llrs[0, :] = y
    
    paths = [first_path]
    decode_node(0, 0, paths, code, L, is_first_round=True, first_round_pms=None, first_round_llrs=None, alpha=999.0, is_flipping=False, flip_layer=-1)
    
    # Check full CRC for surviving paths
    for path in paths:
        u_I = path.bits[code.n, code.info_indices]
        if check_crc(u_I, POLY_CRC5):
            return u_I, True
            
    # Fallback to the best path's estimate if none pass CRC
    if len(paths) > 0:
        return paths[0].bits[code.n, code.info_indices], False
    return np.zeros(code.K, dtype=int), False

# =====================================================================
# 4. SCLF & TS-SCLF (SCLF + SCL-RE Early Termination + Segmented CRC)
# =====================================================================

def run_flipskip(S, i, D1, D2):
    """
    Algorithm 1: Path-Flipping Skip Criterion (FLIPSKIP)
    """
    if i == 0:
        return 0
    kc = 0
    l = S[i]
    for k in range(i):
        if 0 < l - S[k] < D1:
            kc += 1
    if kc >= D2:
        return 1  # Skip
    return 0

def generate_flipping_set(first_round_llrs, code, failed_seg1=True):
    """
    Ranks information layers in ascending order of their absolute LLR magnitude.
    If failed_seg1 is True, constrains the set to layers before or equal to l_seg1.
    """
    info_layers = list(code.info_indices)
    if failed_seg1:
        info_layers = [l for l in info_layers if l <= code.l_seg1]
        
    # Sort layers by LLR magnitude
    sorted_layers = sorted(info_layers, key=lambda l: abs(first_round_llrs.get(l, 999.0)))
    return sorted_layers

def decode_sclf(y, code, L=4, T=10, D1=16, D2=3):
    """
    Standard SCLF Decoder (no early SCL-RE stopping, no Segmented CRC).
    """
    first_path = Path(code.n, code.N)
    first_path.llrs[0, :] = y
    paths = [first_path]
    
    first_round_llrs = {}
    decode_node(0, 0, paths, code, L, is_first_round=True, first_round_pms=None, first_round_llrs=first_round_llrs, alpha=999.0, is_flipping=False, flip_layer=-1)
    
    for path in paths:
        u_I = path.bits[code.n, code.info_indices]
        if check_crc(u_I, POLY_CRC5):
            return u_I, True, 1
            
    S = generate_flipping_set(first_round_llrs, code, failed_seg1=False)
    
    attempts = 1
    for i in range(min(T, len(S))):
        attempts += 1
        fout = run_flipskip(S, i, D1, D2)
        if fout == 1:
            continue
            
        flip_layer = S[i]
        restart_path = Path(code.n, code.N)
        restart_path.llrs[0, :] = y
        restart_paths = [restart_path]
        
        decode_node(0, 0, restart_paths, code, L, is_first_round=False, first_round_pms=None, first_round_llrs=None, alpha=999.0, is_flipping=True, flip_layer=flip_layer)
        
        for path in restart_paths:
            u_I = path.bits[code.n, code.info_indices]
            if check_crc(u_I, POLY_CRC5):
                return u_I, True, attempts
                
    if len(paths) > 0:
        return paths[0].bits[code.n, code.info_indices], False, attempts
    return np.zeros(code.K, dtype=int), False, attempts

def decode_ts_sclf(y, code, L=4, T=10, D1=16, D2=3, alpha=0.5, apply_constraint=True, guard_band=12, use_lra_ast=False, gamma_lra=0.8):
    """
    TS-SCLF Decoder: SCLF + SCL-RE early termination (Eq. 6) + Segmented CRC checks.
    Supports turning the flipping set constraint ON or OFF for debugging.
    """
    # --- Round 0: Initial SCL Decoding ---
    first_path = Path(code.n, code.N)
    first_path.llrs[0, :] = y
    paths = [first_path]
    
    first_round_pms = {}
    first_round_llrs = {}
    
    decode_node(0, 0, paths, code, L, is_first_round=True, first_round_pms=first_round_pms, 
                first_round_llrs=first_round_llrs, alpha=alpha, is_flipping=False, flip_layer=-1, guard_band=guard_band, use_lra_ast=use_lra_ast, gamma_lra=gamma_lra)
    
    # Check if initial SCL decoding succeeded
    for path in paths:
        u_I = path.bits[code.n, code.info_indices]
        if check_crc(u_I, POLY_CRC5):
            return u_I, True, 1, False, 56  # Success, 1 attempt, no seg1 fail, full size
            
    # SCL failed. Determine if it failed Segment 1 CRC (paths list became empty during decoding)
    failed_seg1 = (len(paths) == 0)
    
    # Generate flipping set S (apply constraint based on apply_constraint parameter)
    S = generate_flipping_set(first_round_llrs, code, failed_seg1=(failed_seg1 if apply_constraint else False))
    
    attempts = 1
    # --- Flipping Loop (Attempts 1 to T) ---
    for i in range(min(T, len(S))):
        attempts += 1
        
        # Check FLIPSKIP
        fout = run_flipskip(S, i, D1, D2)
        if fout == 1:
            continue
            
        # Restart SCL with early termination and bit-flipping at S[i]
        flip_layer = S[i]
        restart_path = Path(code.n, code.N)
        restart_path.llrs[0, :] = y
        restart_paths = [restart_path]
        
        decode_node(0, 0, restart_paths, code, L, is_first_round=False, first_round_pms=first_round_pms,
                    first_round_llrs=None, alpha=alpha, is_flipping=True, flip_layer=flip_layer, guard_band=guard_band, use_lra_ast=use_lra_ast, gamma_lra=gamma_lra)
        
        # Check if any path passed final CRC
        for path in restart_paths:
            u_I = path.bits[code.n, code.info_indices]
            if check_crc(u_I, POLY_CRC5):
                return u_I, True, attempts, failed_seg1, len(S)
                
    # Return best available
    if len(paths) > 0:
        return paths[0].bits[code.n, code.info_indices], False, attempts, failed_seg1, len(S)
    return np.zeros(code.K, dtype=int), False, attempts, failed_seg1, len(S)

def decode_adaptive_l_lra_ast(y, code, T=10, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5):
    """
    Adaptive-L LRA-AST Decoder:
    - Initial SCL decoding uses L=4.
    - If it fails, checks the PM range of the surviving L=4 paths at the final info layer.
    - If range < 4.0 (or failed Segment 1 CRC) -> uses L=8 for restarts.
    - If range >= 4.0 -> uses L=4 for restarts.
    """
    # --- Round 0: Initial SCL Decoding with L=4 ---
    L_initial = 4
    first_path = Path(code.n, code.N)
    first_path.llrs[0, :] = y
    paths = [first_path]
    
    first_round_pms = {}
    first_round_llrs = {}
    
    decode_node(0, 0, paths, code, L_initial, is_first_round=True, first_round_pms=first_round_pms, 
                first_round_llrs=first_round_llrs, alpha=alpha, is_flipping=False, flip_layer=-1, guard_band=guard_band, use_lra_ast=use_lra_ast, gamma_lra=gamma_lra)
    
    # Check if initial SCL decoding succeeded
    for path in paths:
        u_I = path.bits[code.n, code.info_indices]
        if check_crc(u_I, POLY_CRC5):
            return u_I, True, 1, 4, False  # Success, 1 attempt, L_restart=4, triggered_L8 = False
            
    # SCL failed. Check if it completed (reached the final info layer) and compute PM range
    triggered_L8 = False
    L_restart = 4
    final_info_layer = code.info_indices[-1]
    if final_info_layer in first_round_pms:
        pms = first_round_pms[final_info_layer]
        # surviving paths range (L_initial = 4)
        surv_pms = pms[:4]
        if len(surv_pms) > 0:
            pm_range = surv_pms[-1] - surv_pms[0]
            if pm_range < 4.0:
                L_restart = 8
                triggered_L8 = True
    else:
        # Failed Segment 1 CRC early termination -> definitely lost, use L=8
        L_restart = 8
        triggered_L8 = True
                
    failed_seg1 = (len(paths) == 0)
    
    # Generate flipping set S
    S = generate_flipping_set(first_round_llrs, code, failed_seg1=(failed_seg1 if apply_constraint else False))
    
    attempts = 1
    # --- Flipping Loop (Attempts 1 to T) ---
    for i in range(min(T, len(S))):
        attempts += 1
        
        # Check FLIPSKIP
        fout = run_flipskip(S, i, D1, D2)
        if fout == 1:
            continue
            
        # Restart SCL with early termination and bit-flipping at S[i]
        flip_layer = S[i]
        restart_path = Path(code.n, code.N)
        restart_path.llrs[0, :] = y
        restart_paths = [restart_path]
        
        decode_node(0, 0, restart_paths, code, L_restart, is_first_round=False, first_round_pms=first_round_pms,
                    first_round_llrs=None, alpha=alpha, is_flipping=True, flip_layer=flip_layer, guard_band=guard_band, use_lra_ast=use_lra_ast, gamma_lra=gamma_lra)
        
        # Check if any path passed final CRC
        for path in restart_paths:
            u_I = path.bits[code.n, code.info_indices]
            if check_crc(u_I, POLY_CRC5):
                return u_I, True, attempts, L_restart, triggered_L8
                
    # Return best available
    if len(paths) > 0:
        return paths[0].bits[code.n, code.info_indices], False, attempts, L_restart, triggered_L8
    return np.zeros(code.K, dtype=int), False, attempts, L_restart, triggered_L8

def decode_bounded_lra_ast(y, code, L=4, T=6, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5):
    """
    Bounded LRA-AST Decoder:
    LRA-AST with a hard limit of T=6 attempts (restarts) per frame.
    """
    return decode_ts_sclf(
        y, code, L=L, T=T, D1=D1, D2=D2, alpha=alpha, 
        apply_constraint=apply_constraint, guard_band=guard_band, 
        use_lra_ast=use_lra_ast, gamma_lra=gamma_lra
    )

def decode_alas_sclf(y, code, T=10, D1=16, D2=3, alpha=1.0, guard_band=8):
    """
    ALAS-SCLF: Adaptive-L Asymmetric-Segmented SCLF.
    - Decodes with L=4 in the first round.
    - If initial decoding fails, dynamically elevates restart list size to L=8
      if surviving path metric range < 4.0 or Segment 1 CRC failed.
    - Restricts flipping candidates to Segment 1 or Segment 2 based on Segment 1 CRC.
    - Restarts use early termination with fixed alpha.
    """
    # Round 0: Initial SCL with L=4
    L_initial = 4
    first_path = Path(code.n, code.N)
    first_path.llrs[0, :] = y
    paths = [first_path]
    
    first_round_pms = {}
    first_round_llrs = {}
    
    decode_node(0, 0, paths, code, L_initial, is_first_round=True, first_round_pms=first_round_pms, 
                first_round_llrs=first_round_llrs, alpha=alpha, is_flipping=False, flip_layer=-1, 
                guard_band=guard_band, use_lra_ast=False)
    
    # Check if initial run succeeded
    for path in paths:
        u_I = path.bits[code.n, code.info_indices]
        if check_crc(u_I, POLY_CRC5):
            return u_I, True, 1
            
    # Check if we should trigger L=8 for restarts
    L_restart = 4
    final_info_layer = code.info_indices[-1]
    if final_info_layer in first_round_pms:
        pms = first_round_pms[final_info_layer]
        surv_pms = pms[:4]
        if len(surv_pms) > 0:
            pm_range = surv_pms[-1] - surv_pms[0]
            if pm_range < 4.0:
                L_restart = 8
    else:
        # Failed Segment 1 CRC early termination -> use L=8
        L_restart = 8
        
    failed_seg1 = (len(paths) == 0)
    
    # Generate flipping set with Segment Constraint
    info_layers = list(code.info_indices)
    if failed_seg1:
        info_layers = [l for l in info_layers if l <= code.l_seg1]
        
    S = sorted(info_layers, key=lambda l: abs(first_round_llrs.get(l, 999.0)))
    
    attempts = 1
    for i in range(min(T, len(S))):
        attempts += 1
        
        fout = run_flipskip(S, i, D1, D2)
        if fout == 1:
            continue
            
        flip_layer = S[i]
        restart_path = Path(code.n, code.N)
        restart_path.llrs[0, :] = y
        restart_paths = [restart_path]
        
        decode_node(0, 0, restart_paths, code, L_restart, is_first_round=False, first_round_pms=first_round_pms,
                    first_round_llrs=None, alpha=alpha, is_flipping=True, flip_layer=flip_layer, 
                    guard_band=guard_band, use_lra_ast=False)
        
        for path in restart_paths:
            u_I = path.bits[code.n, code.info_indices]
            if check_crc(u_I, POLY_CRC5):
                return u_I, True, attempts
                
    if len(paths) > 0:
        return paths[0].bits[code.n, code.info_indices], False, attempts
    return np.zeros(code.K, dtype=int), False, attempts

# =====================================================================
# 5. Comparative Debug Simulation Sweep
# =====================================================================

def encode_segmented_crc_generic(info_payload, K):
    """
    Generic segmented CRC encoder for K information bits:
    - Segment 1: first l_seg payload bits + 3-bit CRC1
    - Segment 2: remaining K - 8 - l_seg payload bits + 5-bit CRC2
    Total info bits = K.
    """
    l_seg = (K - 8) // 2
    u_I = np.zeros(K, dtype=int)
    # Segment 1: first l_seg bits
    u_I[0:l_seg] = info_payload[0:l_seg]
    u_I[l_seg : l_seg + 3] = compute_crc(u_I[0:l_seg], POLY_CRC3)
    # Segment 2: next K - 8 - l_seg bits
    u_I[l_seg + 3 : K - 5] = info_payload[l_seg : K - 8]
    u_I[K - 5 : K] = compute_crc(u_I[0 : K - 5], POLY_CRC5)
    return u_I

if __name__ == "__main__":
    print("=================================================================")
    print("                LRA-AST RATE GENERALIZATION SWEEP                ")
    print("=================================================================")
    
    np.random.seed(42)  # Set seed for reproducibility
    
    rates_configs = [
        {"R": 0.25,   "K": 32, "l_seg": 12},
        {"R": 0.375,  "K": 48, "l_seg": 20},
        {"R": 0.4375, "K": 56, "l_seg": 24},
        {"R": 0.5,    "K": 64, "l_seg": 28},
        {"R": 0.625,  "K": 80, "l_seg": 36}
    ]
    
    N = 128
    snr_db = 1.0
    num_trials = 1000
    
    print(f"Running rate generalization sweep: {num_trials} trials at Eb/N0 = {snr_db:.1f} dB.")
    print(f"LRA-AST parameters: alpha_0 = 1.5, gamma = 0.5 (optimized at R = 0.4375)\n")
    
    results = {}
    
    for config in rates_configs:
        R = config["R"]
        K = config["K"]
        l_seg = config["l_seg"]
        
        print(f"Evaluating Rate R = {R:.4f} (N = {N}, K = {K}, l_seg = {l_seg})...")
        
        code = PolarCode(N=N, K=K, l_seg=l_seg)
        code.early_termination_count = 0
        
        # Calculate noise sigma for this rate
        ebno = 10**(snr_db / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * R * ebno))
        
        sclf_fer = 0
        sclf_attempts = 0.0
        
        lra_fer = 0
        lra_attempts = 0.0
        
        for trial in range(num_trials):
            # Generate random payload of size K - 8
            payload = np.random.randint(0, 2, K - 8)
            u_I = encode_segmented_crc_generic(payload, K)
            
            # Encode
            x = code.encode(u_I)
            
            # BPSK + AWGN
            s = 1 - 2 * x
            noise = np.random.normal(0, sigma, N)
            r = s + noise
            y = 2 * r / (sigma**2)
            
            # 1. SCLF Decoder
            dec_sclf, success_sclf, attempts_sclf = decode_sclf(y, code, L=4, T=10, D1=16, D2=3)
            if not success_sclf or not np.array_equal(dec_sclf, u_I):
                sclf_fer += 1
            sclf_attempts += attempts_sclf
            
            # 2. LRA-AST Decoder (using alpha_0=1.5, gamma=0.5)
            dec_lra, success_lra, attempts_lra, _, _ = decode_ts_sclf(
                y, code, L=4, T=10, D1=16, D2=3, alpha=1.5, 
                apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
            )
            if not success_lra or not np.array_equal(dec_lra, u_I):
                lra_fer += 1
            lra_attempts += attempts_lra
            
        results[R] = {
            "sclf_fer": sclf_fer / num_trials,
            "sclf_attempts": sclf_attempts / num_trials,
            "lra_fer": lra_fer / num_trials,
            "lra_attempts": lra_attempts / num_trials,
            "et_fires": code.early_termination_count
        }
        
        print(f"  SCLF:    FER = {results[R]['sclf_fer']:.5f}, Avg Attempts = {results[R]['sclf_attempts']:.3f}")
        print(f"  LRA-AST: FER = {results[R]['lra_fer']:.5f}, Avg Attempts = {results[R]['lra_attempts']:.3f} (ET Fires = {results[R]['et_fires']})")
        print("-" * 65)
        
    print("\n" + "="*85)
    print("                         RATE GENERALIZATION SWEEP SUMMARY")
    print("="*85)
    print(f"{'Rate (R)':<10}{'K':<6}{'SCLF FER':<15}{'SCLF Avg Att':<15}{'LRA-AST FER':<15}{'LRA-AST Avg Att':<15}{'ET Fires':<10}")
    print("-"*85)
    for config in rates_configs:
        R = config["R"]
        res = results[R]
        print(f"{R:<10.4f}{config['K']:<6}{res['sclf_fer']:<15.5f}{res['sclf_attempts']:<15.3f}{res['lra_fer']:<15.5f}{res['lra_attempts']:<15.3f}{res['et_fires']:<10}")
    print("="*85)
