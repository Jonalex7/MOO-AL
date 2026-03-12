from typing import List, Optional

import numpy as np
from scipy.stats import norm

from active_learning.eier import select_eier_index
from utils.data import normalize_array


class AcquisitionStrategy:
    """
    This class holds methods for acquisition functions such as
    U-function, EFF, and multi-objective Pareto-based selection (including reliability adaptation).
    """
    def __init__(
        self,
        acquisition_strategy: str, # 'u', 'eff', or 'moo'
        moo_method: Optional[str] = None, # 'knee', 'compromise', 'reliability', or 'linear_decay'
        N_it: int = 2, # Number of iterations to consider for moving average in reliability method
        delta_P0: float = 0.2, # (0,1) threshold of relative difference at which gamma=0.5
        k_balance: float = 40,  # Positive constant controlling how quickly gamma transition from 0 to 1
        input_dim: int = 2,
        pareto_metrics: bool = False, # If True, returns Pareto front and selected indices
        eps_start: float = 1.0,     # start with exploration emphasis
        eps_end: float = 0.0,       # end with exploitation emphasis
        eps_T: int = 100,           # number of calls to decay over
        portfolio_lambda: float = 2.0,   # Hedge balance (lambda)
        portfolio_delta: float = 0.7,    # Memory factor (delta)
        batch_size_acq: int = 500,
        n_z_mc: int = 64,
        jitter_stddev: float = 1e-8,
        local_mis_topk: int = 3000,
        debug_acq: bool = False,
        eier_num_workers: int = 1,
        z_chunk_size: int = 64,
        n_mcs_eier_int: Optional[int] = None,
    ):
        self.strategy = acquisition_strategy.lower().strip()
        self.pareto_metrics = pareto_metrics

        if self.strategy == "moo":
            if moo_method not in ("knee", "compromise", "reliability", "linear_decay"):
                raise ValueError(
                    "`moo_method` must be 'knee', 'compromise', 'reliability', or 'linear_decay'"
                )
            self.moo_method = moo_method
            # Initialize reliability parameters only when using moo_reliability
            if self.moo_method == "reliability":
                self.N_it = N_it
                self.delta_P0 = delta_P0
                self.k_balance = k_balance
                self.input_dim = input_dim
                self.Pf_prev = 0.0
                self.delta_Pf_buffer: List[float] = []
            # Linear-decay schedule state (used by linear_decay method).
            if self.moo_method == "linear_decay":
                self.eps_start = float(eps_start)
                self.eps_end   = float(eps_end)
                self.eps_T     = int(eps_T)
                self._eps_t    = 0  # internal call counter

        # Portfolio strategy initialization
        if self.strategy == "portfolio":
            # order of arms (must match the call sequence below)
            self._arms: List[str] = ["u", "eff", "erf", "reif", "reif2"]
            self._K = len(self._arms)

            # Hedge state: total rewards G_i and probabilities p_i
            self._G = np.zeros(self._K, dtype=np.float64)                 # totals
            self._p = np.full((self._K,), 1.0/self._K, dtype=np.float64)  # probs

            self._lambda = float(portfolio_lambda)
            self._delta  = float(portfolio_delta)

            # tracking which arm selected each iteration
            self.portfolio_history: List[str] = []
            self.rewards_history: List[str] = []
            # counts per arm
            self.portfolio_counts = {a: 0 for a in self._arms}

        if self.strategy == "eier":
            self.batch_size_acq = int(batch_size_acq)
            self.n_z_mc = int(n_z_mc)
            self.jitter_stddev = float(jitter_stddev)
            self.local_mis_topk = int(local_mis_topk)
            self.debug_acq = bool(debug_acq)
            self.eier_num_workers = max(1, int(eier_num_workers))
            self.z_chunk_size = max(1, int(z_chunk_size))
            self.n_mcs_eier_int = None if n_mcs_eier_int is None else int(n_mcs_eier_int)

    def get_indices(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        input_candidates: Optional[np.ndarray] = None,
        model_gp = None,
        candidate_pool: Optional[np.ndarray] = None,
        z_seed: Optional[int] = None,
        n_samples: int = 1,
        skip_indices: Optional[List[int]] = None,
        constant: float = 2.0,
        pf_estimate: Optional[float] = None
    ) -> List[int]:

        # ---------- MOO-based selection  ----------
        if self.strategy == "moo":
            pareto, selected_indices, p_min, p_max = self.get_moo(
                mean_prediction,
                std_prediction,
                self.moo_method,
                pf_estimate=pf_estimate
            )
            if self.pareto_metrics:
                return pareto, selected_indices, p_min, p_max
            else:
                return selected_indices

        # ---------- Non-MOO strategies (all handled uniformly) ----------
        # Portfolio: always select *one* sample via the portfolio step
        if self.strategy == "portfolio":
            if n_samples != 1:
                raise ValueError("Portfolio strategy currently supports n_samples=1 only.")

            idx = self._portfolio_step(
                mean_prediction=mean_prediction,
                std_prediction=std_prediction,
                input_candidates=input_candidates,
                skip_indices=skip_indices,
            )
            selected_indices = [idx]

        # U-based selection
        elif self.strategy == "u":
            selected_indices = self._u_function(
                mean_prediction,
                std_prediction,
                n_samples,
                skip_indices,
            )

        # EFF-based selection
        elif self.strategy == "eff":
            selected_indices = self._eff_function(
                mean_prediction,
                std_prediction,
                n_samples,
                skip_indices,
                constant,
            )

        # ERF-based selection
        elif self.strategy == "erf":
            selected_indices = self._erf_function(
                mean_prediction,
                std_prediction,
                n_samples,
                skip_indices,
            )

        # REIF-based selection
        elif self.strategy == "reif":
            selected_indices = self._reif_function(
                mean_prediction,
                std_prediction,
                n_samples,
                skip_indices,
            )

        # REIF2-based selection
        elif self.strategy == "reif2":
            selected_indices = self._reif2_function(
                mean_prediction,
                std_prediction,
                input_candidates,
                n_samples,
                skip_indices,
            )

        # EIER-based selection
        elif self.strategy == "eier":
            selected_indices = self._eier_function(
                model_gp,
                candidate_pool,
                mean_prediction,
                std_prediction,
                n_samples,
                z_seed,
                skip_indices,
            )

        else:
            raise ValueError(f"Unknown acquisition strategy '{self.strategy}'")

        # ---------- Common Pareto-metrics ----------
        if self.pareto_metrics:
            mean_pred_norm = normalize_array(np.abs(mean_prediction))
            std_pred_norm  = normalize_array(std_prediction)
            pareto, _, _, _, _, _, _, p_min, p_max = self.compute_pareto_front(mean_pred_norm, std_pred_norm)
            return pareto, selected_indices, p_min, p_max
        else:
            return selected_indices

    def _eier_function(
        self,
        model_gp,
        candidate_pool: Optional[np.ndarray],
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        n_samples: int,
        z_seed: Optional[int] = None,
        skip_indices: Optional[List[int]] = None,
    ) -> List[int]:
        if n_samples != 1:
            raise ValueError("EIER strategy currently supports n_samples=1 only.")
        if model_gp is None:
            raise ValueError("`model_gp` is required for EIER selection.")
        if candidate_pool is None:
            raise ValueError("`candidate_pool` is required for EIER selection.")
        if z_seed is None:
            z_seed = 0

        selected_index = select_eier_index(
            model_gp=model_gp,
            candidate_pool=candidate_pool,
            n_mcs_eier_int=self.n_mcs_eier_int,
            batch_size_acq=self.batch_size_acq,
            n_z_mc=self.n_z_mc,
            jitter_stddev=self.jitter_stddev,
            local_mis_topk=self.local_mis_topk,
            z_seed=int(z_seed),
            debug_acq=self.debug_acq,
            skip_indices=skip_indices,
            mean_prediction=mean_prediction,
            std_prediction=std_prediction,
            num_workers=self.eier_num_workers,
            z_chunk_size=self.z_chunk_size,
        )
        return [selected_index]

    def _u_function(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        n_samples: int, # Number of samples to select
        skip_indices: Optional[List[int]] # Indices to skip in the pool
    ) -> List[int]:
        mu = np.asarray(mean_prediction, dtype=np.float64).squeeze()
        sig = np.asarray(std_prediction, dtype=np.float64).squeeze()
        u = np.abs(mu) / sig
        if skip_indices is not None:
            u = u.copy()
            u[skip_indices] = float('inf')
        u_idx = np.argsort(u)[:n_samples]
        return u_idx.tolist()

    def _eff_function(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        n_samples: int, # Number of samples to select
        skip_indices: Optional[List[int]], # Indices to skip in the pool
        constant: float = 2.0
    ) -> List[int]:
        mean_prediction = np.asarray(mean_prediction, dtype=np.float64).squeeze()
        std_prediction = np.asarray(std_prediction, dtype=np.float64).squeeze()
        eps = constant * std_prediction
        eff = (
            mean_prediction
            * (
                2 * norm.cdf(-mean_prediction / std_prediction)
                - norm.cdf(-(eps + mean_prediction) / std_prediction)
                - norm.cdf((eps - mean_prediction) / std_prediction)
            )
            - std_prediction
            * (
                2 * norm.pdf(-mean_prediction / std_prediction)
                - norm.pdf(-(eps + mean_prediction) / std_prediction)
                - norm.pdf((eps - mean_prediction) / std_prediction)
            )
            + eps
            * (
                norm.cdf((eps - mean_prediction) / std_prediction)
                - norm.cdf((-eps - mean_prediction) / std_prediction)
            )
        )
        if skip_indices is not None:
            eff = eff.copy()
            eff[skip_indices] = float('-inf')
        eff_idx = np.argsort(eff)[-n_samples:][::-1]
        return eff_idx.tolist()

    def _erf_function(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        n_samples: int,                         # number of samples to select
        skip_indices: Optional[List[int]] = None  # indices to skip in the pool
    ) -> List[int]:
        """
        Expected Risk Function (ERF) sampling criterion.
        Selects the indices corresponding to the largest ERF values.
        Reference: Yang et al. (2015) ALK-HRA with Expected Risk Function.
        """
        mu = np.asarray(mean_prediction, dtype=np.float64).squeeze()
        sig = np.asarray(std_prediction, dtype=np.float64).squeeze()

        sgn = np.where(mu >= 0.0, 1.0, -1.0)
        z = mu / sig

        phi = norm.pdf(z)
        Phi_neg = norm.cdf(-sgn * z)

        erf_val = -sgn * mu * Phi_neg + sig * phi  # larger = higher expected risk

        if skip_indices is not None:
            erf_val = erf_val.copy()
            erf_val[skip_indices] = float('-inf')

        idx = np.argsort(erf_val)[-n_samples:][::-1]
        return idx.tolist()

    def _reif_function(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        n_samples: int,
        skip_indices: Optional[List[int]] = None,
        w: float = 2.0,  # as suggested in the paper
    ) -> List[int]:
        """
        REIF selector (maximize): REIF = w*σ - E[|ĝ|],
        where E[|N(μ, σ²)|] = σ*sqrt(2/pi)*exp(-0.5*(μ/σ)^2) + μ*(1 - 2*Φ(μ/σ))
        Reference: Zhang, Wang & Sørensen (2019), RESS. REIF/REIF2. 
        """
        mu = np.asarray(mean_prediction, dtype=np.float64).squeeze()
        sig = np.asarray(std_prediction, dtype=np.float64).squeeze()

        # compute beta = mu/sig
        beta_np = (mu / sig)
        Phi_beta = norm.cdf(beta_np)

        # folded-normal expectation E|ĝ|
        term_var = (w - np.sqrt(2.0/np.pi) * np.exp(-0.5 * (beta_np)**2))
        term_mean = mu * (1.0 - 2.0 * Phi_beta)

        reif = term_mean + sig * term_var # larger is better

        if skip_indices is not None:
            reif = reif.copy()
            reif[skip_indices] = float('-inf')

        idx = np.argsort(reif)[-n_samples:][::-1]
        return idx.tolist()
    
    def _reif2_function(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        input_candidates: np.ndarray,   # f_X(x) evaluated at each candidate (same shape as mu)
        n_samples: int,
        skip_indices: Optional[List[int]] = None,
        w: float = 2.0,
    ) -> List[int]:
        """
        REIF2 selector (maximize): REIF2 = REIF * f_X(x).
        Same REIF core as above, with multiplicative modulation by the input PDF.
        Reference: Zhang, Wang & Sørensen (2019), RESS. REIF/REIF2.
        """
        mu = np.asarray(mean_prediction, dtype=np.float64).squeeze()
        sig = np.asarray(std_prediction, dtype=np.float64).squeeze()
        fx = self.std_normal_pdf_product(input_candidates)

        # compute beta = mu/sig
        beta_np = (mu / sig)
        Phi_beta = norm.cdf(beta_np)

        # folded-normal expectation E|ĝ|
        term_var = (w - np.sqrt(2.0/np.pi) * np.exp(-0.5 * (beta_np)**2))
        term_mean = mu * (1.0 - 2.0 * Phi_beta)

        reif = term_mean + sig * term_var # larger is better
        reif2 = reif * fx

        if skip_indices is not None:
            reif2 = reif2.copy()
            reif2[skip_indices] = float('-inf')

        idx = np.argsort(reif2)[-n_samples:][::-1]
        return idx.tolist()

    def get_moo(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        method: Optional[str] = None,  # 'knee', 'compromise', 'reliability', or 'linear_decay'
        pf_estimate: Optional[float] = None, # Current Pf estimate for reliability method (if applicable)
    ) -> List[int]:
        """
        Multi-objective selection via Pareto front.
        method: 'knee', 'compromise', 'reliability', or 'linear_decay'
        """
        # Compute the Pareto front
        mean_pred_norm = normalize_array(np.abs(mean_prediction))
        std_pred_norm = normalize_array(std_prediction)
        pareto_front, pareto_front_indices, _, knee_idx, _, comp_idx, _, p_min, p_max = self.compute_pareto_front(
            mean_pred_norm, std_pred_norm
        )
        # select the knee point, compromised point, or reliability point
        if method == 'knee':
            return pareto_front, [int(knee_idx)], p_min, p_max
        elif method == 'compromise':
            return pareto_front, [int(comp_idx)], p_min, p_max
        elif method == 'reliability':
            moo_pareto_index = self.get_moo_reliability(pareto_front=pareto_front, pf_estimate=pf_estimate)
            return pareto_front, [int(pareto_front_indices[moo_pareto_index])], p_min, p_max
        elif method == 'linear_decay':
            pos_on_front = self.get_moo_linear_decay_euclidean(pareto_front)
            return pareto_front, [int(pareto_front_indices[pos_on_front])], p_min, p_max
        else:
            raise ValueError(f"Unknown MO pareto strategy: {method}")

    def compute_pareto_front(
        self,
        mean_pred: np.ndarray,
        std_pred: np.ndarray
    ):
        mean_pred = np.asarray(mean_pred, dtype=np.float64).squeeze()
        std_pred = np.asarray(std_pred, dtype=np.float64).squeeze()

        # Negate mean_pred for minimization via maximization logic
        objectives = np.stack([-mean_pred, std_pred], axis=1)
        is_pareto = np.ones(objectives.shape[0], dtype=bool)
        for i, pt in enumerate(objectives):
            if is_pareto[i]:
                dominated = np.all(objectives <= pt, axis=1) & np.any(objectives < pt, axis=1)
                is_pareto[dominated] = False
        indices = np.flatnonzero(is_pareto)
        # -----------------------------------
        front = objectives[is_pareto]
        if indices.size == 1:
            order = np.argsort(front[:, 0])
            indices = indices[order]
            front = front[order]

            knee_pt = front[0].copy()
            comp_pt = front[0].copy()
            ideal_pt = front[0].copy()
            pareto_min_vals = front.copy()
            pareto_max_vals = front.copy()

            return front, indices, knee_pt, int(indices[0]), comp_pt, int(indices[0]), ideal_pt, pareto_min_vals, pareto_max_vals
        
        # Normalize the Pareto front to [0, 1]
        pareto_min_vals = np.min(front, axis=0, keepdims=True) # Minimum values for normalization
        pareto_max_vals = np.max(front, axis=0, keepdims=True) # Maximum values for normalization
        denom = pareto_max_vals - pareto_min_vals
        denom[denom == 0.0] = 1.0
        front = (front - pareto_min_vals) / denom
        # Sort by first objective
        order = np.argsort(front[:,0])
        front = front[order]
        indices = indices[order]
        knee_pt, knee_idx = self.calculate_knee_point(front)
        comp_pt, comp_idx, ideal_pt = self.calculate_compromised_point(front)
        
        return front, indices, knee_pt, int(indices[knee_idx]), comp_pt, int(indices[comp_idx]), ideal_pt, pareto_min_vals, pareto_max_vals

    def calculate_knee_point(self, pareto_front: np.ndarray):
        p1, p2 = pareto_front[0], pareto_front[-1]
        line = p2 - p1
        line = line / np.linalg.norm(line)
        dists = np.zeros(pareto_front.shape[0], dtype=np.float64)
        for i, pt in enumerate(pareto_front):
            vec = pt - p1
            proj = p1 + np.dot(vec, line) * line
            dists[i] = np.linalg.norm(pt - proj)
        idx = int(np.argmax(dists))
        return pareto_front[idx], idx

    def calculate_compromised_point(self, pareto_front: np.ndarray):
        ideal = np.max(pareto_front, axis=0)
        dists = np.linalg.norm(pareto_front - ideal, axis=1)
        idx = int(np.argmin(dists))
        return pareto_front[idx], idx, ideal
        
    def logistic_gamma(self, delta_P, delta_P0=0.2, k=40):
        gamma_max = 0.9
        gamma = gamma_max*(1 / (1 + np.exp(-k * (delta_P - delta_P0))))
        return gamma

    def std_normal_pdf_product(self, input_candidates: np.ndarray) -> np.ndarray:
        pdf = norm.pdf(input_candidates)                 # (N, D)
        pdf_joint = pdf.prod(axis=1)              # independent product
        return pdf_joint

    def get_moo_reliability(self, pareto_front, pf_estimate):
        # Checking Pf rel. difference to choose gamma behaviour
        Pf_current = pf_estimate

        # Calculate the relative difference from the previous Pf
        if self.Pf_prev != 0:
            delta_Pf = abs(Pf_current - self.Pf_prev) / self.Pf_prev
        else:
            delta_Pf = 1e2  # Handle division by zero

        # Update the buffer with the latest delta_Pf
        self.delta_Pf_buffer.append(delta_Pf)
        if len(self.delta_Pf_buffer) > self.N_it:
            self.delta_Pf_buffer.pop(0)  # Keep only the last N values

        delta_avg = float(np.mean(self.delta_Pf_buffer))
        # # Extract mean predictions and standard deviations
        normalized_mean = pareto_front[:, 0]
        normalized_std = pareto_front[:, 1]
        
        gamma = self.logistic_gamma(delta_avg, delta_P0=self.delta_P0, k=self.k_balance)
        print(f'delta_pf_avg: {delta_avg:.3f}, gamma_log: {gamma:.3f} \n')
        # Update previous Pf for next iteration
        self.Pf_prev = Pf_current

        # ------------------------------------------------------------------
        # Euclidean-compromise scalarization 
        # ------------------------------------------------------------------
        # Ideal point at (1, 1)
        delta_mean = 1.0 - normalized_mean
        delta_std = 1.0 - normalized_std

        # gamma controls exploration vs exploitation:
        #   gamma -> more exploration
        #   gamma -> more exploitation
        w_mean = 1.0 - gamma   # weight on mean term
        w_std = gamma          # weight on std term

        # Weighted squared distance to ideal point (no need sqrt for argmin)
        dist_sq = w_mean * (delta_mean ** 2) + w_std * (delta_std ** 2)

        arg_min = int(np.argmin(dist_sq))
        return arg_min

    def _linear_decay_value(self) -> float:
        """Linear decay parameter in [eps_start -> eps_end] over eps_T calls."""
        if self.eps_T <= 0:
            return self.eps_end
        frac = min(1.0, self._eps_t / self.eps_T)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def reset_linear_decay_schedule(self):
        """Optional: call this if you want to restart the linear-decay schedule."""
        self._eps_t = 0

    def _portfolio_step(
        self,
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        input_candidates: Optional[np.ndarray],  # needed for REIF2
        skip_indices: Optional[List[int]] = None
    ) -> int:
        """
        One Hedge update + selection:
        - Each arm proposes its best idx
        - Reward r_i = -|mu(best_i)|
        - Totals G_i <- δ G_i + r_i
        - p_i = softmax(λ * normalized(G))
        - Sample one arm by p_i and return its idx
        """
        if input_candidates is None:
            # Only REIF2 needs candidates; we still require it here to keep interface simple
            raise ValueError("`input_candidates` (N,D) is required (for REIF2 portfolio arm).")
        mu = np.asarray(mean_prediction, dtype=np.float64).squeeze()
        sig = np.asarray(std_prediction, dtype=np.float64).squeeze()

        # 1) Each arm proposes best index (reuses your existing functions)
        arm_best: List[int] = []
        arm_best.append(self._u_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._eff_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._erf_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._reif_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._reif2_function(mu, sig, input_candidates, n_samples=1, skip_indices=skip_indices)[0])

        # 2) rewards r_i = -|mu(best_i)|
        mu_best = mu[np.asarray(arm_best, dtype=int)]
        rewards = -np.abs(mu_best).astype(np.float64)
        # 3) totals update with memory
        self._G = self._delta * self._G + rewards

        # 4) probabilities via softmax on normalized totals
        Gmax = float(self._G.max())
        Gmin = float(self._G.min())
        if Gmax == Gmin:
            self._p = np.full((self._K,), 1.0 / self._K, dtype=np.float64)
        else:
            q = (self._G - Gmax) / (Gmax - Gmin)   # in [-1,0]
            logits = self._lambda * q
            m = float(logits.max())
            expv = np.exp(logits - m)
            self._p = expv / expv.sum()
        # sample one arm and return its proposed index
        arm_idx = int(np.random.choice(self._K, p=self._p))
        chosen_idx = int(arm_best[arm_idx])

        # tracking
        chosen_arm = self._arms[arm_idx]
        self.portfolio_history.append(chosen_arm)
        self.portfolio_counts[chosen_arm] += 1
        self.rewards_history.append(self._G.tolist())
        return chosen_idx

    def get_moo_linear_decay_euclidean(self, pareto_front: np.ndarray) -> int:
        """
        Linear decay via Euclidean-compromise scalarization on the current Pareto front.

        We assume pareto_front[:, 0] and [:, 1] are 'higher is better' scores
        (e.g. something like [-mean_norm, std_norm] upstream).

        We:
        1) Min-max normalize each objective on the current front to [0,1].
        2) Define an ideal point at (1,1).
        3) Compute weighted distance to the ideal point:
            d^2 = w * δf_mu^2 + (1 - w) * δf_sigma^2
            where δf_mu = 1 - f_mu_norm, δf_sigma = 1 - f_sigma_norm.
        4) Select argmin d^2.
        """

        K = pareto_front.shape[0]
        if K == 0:
            raise ValueError("Empty Pareto front.")
        if K == 1:
            self._eps_t += 1
            return 0

        pf = pareto_front

        # 1) Per-column min-max normalization on the current Pareto front
        col_min = np.min(pf, axis=0)
        col_max = np.max(pf, axis=0)
        denom = np.maximum(col_max - col_min, 1e-12)
        pf_norm = (pf - col_min) / denom  # shape (K, 2), values in [0,1]

        # 2) Distances to the ideal point (1,1)
        delta_mu    = 1.0 - pf_norm[:, 0]   
        delta_sigma = 1.0 - pf_norm[:, 1]  

        # Linear schedule:
        # value ~ 1 => emphasize exploration
        # value ~ 0 => emphasize exploitation
        decay_value = float(self._linear_decay_value())
        w = 1.0 - decay_value           # w in [0,1]

        # Weighted squared distance (no need to take sqrt: argmin is the same)
        dist_sq = w * (delta_mu ** 2) + (1.0 - w) * (delta_sigma ** 2)

        idx_on_front = int(np.argmin(dist_sq))
        self._eps_t += 1
        return idx_on_front
