import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#class HypergraphFusion(nn.Module):
#    """
#    Spectral Hypergraph Fusion Module.
#
#    Learns to fuse per‐node feature matrix X via K learnable hypergraph kernels.
#
#    Args:
#        input_dim (int):  Dimensionality of each node’s feature vector (e.g. 3d).
#        output_dim (int): Dimensionality of output fused features.
#        num_kernels (int): Number of hypergraph kernels (heads) to ensemble.
#        use_attention (bool): If True, apply node‐wise attention gating over the fused graph.
#    """
#
#    def __init__(self, input_dim: int, output_dim: int,
#                 num_kernels: int = 4, use_attention: bool = False):
#        super().__init__()
#        self.K = num_kernels
#        self.use_attention = use_attention
#
#        # --- learnable kernels: each defines a spectral filter via bilinear form ---
#        # L_k : (input_dim × input_dim) per head
#        self.L_k = nn.ParameterList([
#            nn.Parameter(torch.randn(input_dim, input_dim) * 0.02)
#            for _ in range(self.K)
#        ])
#        # α: head weights
#        self.alpha = nn.Parameter(torch.ones(self.K))
#
#        # Optional node‐wise attention gate: maps each node feature to a scalar
#        if self.use_attention:
#            self.att_mlp = nn.Sequential(
#                nn.Linear(input_dim, input_dim),
#                nn.Tanh(),
#                nn.Linear(input_dim, 1),
#                nn.Sigmoid()
#            )
#
#        # Final projection after spectral fusion
#        self.proj = nn.Linear(input_dim, output_dim)
#
#    def forward(self, X: torch.Tensor) -> torch.Tensor:
#        """
#        Args:
#            X (Tensor): Node features, shape (N, input_dim)
#
#        Returns:
#            Z (Tensor): Fused node features, shape (N, output_dim)
#        """
#        N, D = X.shape  # N nodes, D=input_dim
#        H_heads = []
#        # Build each hypergraph head
#        for L in self.L_k:
#            # Compute adjacency via bilinear filter + row‐softmax:
#            # H_k = softmax( X L X^T )  with shape (N, N)
#            A = X @ L @ X.t()  # (N, N)
#            H_k = F.softmax(A / math.sqrt(D), dim=1)  # normalize rows
#            H_heads.append(H_k)
#
#        # Stack to (K, N, N) and ensemble via α
#        H_stack = torch.stack(H_heads, dim=0)  # (K, N, N)
#        w = F.softmax(self.alpha, dim=0)  # (K,)
#        H = torch.einsum('kij,k->ij', H_stack, w)  # (N, N)
#
#        # Optionally gate each row by node‐wise attention
#        if self.use_attention:
#            # att: (N, 1)
#            att = self.att_mlp(X)
#            H = H * att  # broadcasting
#
#        # Spectral message passing: Z_temp = H @ X
#        Z_temp = H @ X  # (N, D)
#
#        # Project to desired output dim
#        Z = self.proj(Z_temp)  # (N, output_dim)
#        return Z

class HypergraphFusion(nn.Module):
    """
    Spectral Hypergraph Fusion (with kernel blend).

    Pipeline:
      1) Learn K bilinear kernels:  A_k = softmax( X L_k X^T / tau_row / sqrt(D) )  (N×N)
      2) Convert rows of A_k to soft hyperedges (per-head, per-sender) with top-k sparsification
      3) Build normalized hypergraph Laplacian: L = I - Dv^{-1/2} B W De^{-1} B^T Dv^{-1/2}
      4) Apply Chebyshev spectral filter T=0..T (with λ_max via power iteration), then nonlinearity
      5) Residual blend with kernel diffusion H_fused @ X using learnable rho
      6) Final linear projection to output_dim
    """

    def __init__(self, input_dim: int, output_dim: int,
                 num_kernels: int = 4,
                 topk: int = 16,
                 cheb_T: int = 2,
                 row_temp: float = 1.0,
                 use_attention: bool = False,
                 lap_dropout_p: float = 0.0):
        super().__init__()
        self.K = num_kernels
        self.topk = topk
        self.cheb_T = cheb_T
        self.row_temp = row_temp
        self.use_attention = use_attention
        self.lap_dropout_p = lap_dropout_p

        # --- learnable bilinear kernels & mixing weights ---
        self.L_k = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.02)
            for _ in range(self.K)
        ])
        self.alpha = nn.Parameter(torch.ones(self.K))

        # Optional node-wise gate (sender & receiver)
        if self.use_attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            )

        # Chebyshev coefficients θ_0..θ_T (shared across channels; broadcast over feature dim)
        self.theta = nn.Parameter(torch.randn(self.cheb_T + 1, input_dim) * 0.02)

        # Residual blend parameter rho \in (0,1)
        self.rho_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid -> ~0.5 if init at 0

        # Final projection
        self.proj = nn.Linear(input_dim, output_dim)

        # Buffers for diagnostics
        self.register_buffer("last_lambda_max", torch.tensor(0.0), persistent=False)

    # ---------- helpers ----------
    @staticmethod
    def _row_softmax(A: torch.Tensor, tau: float = 1.0):
        return F.softmax(A / max(tau, 1e-6), dim=1)

    def _build_kernel_stack(self, X: torch.Tensor):
        """Return H_stack:(K,N,N) and H_fused:(N,N) row-stochastic kernels."""
        N, D = X.shape
        heads = []
        for L in self.L_k:
            A = X @ L @ X.t() / math.sqrt(D)
            Hk = self._row_softmax(A, self.row_temp)  # (N,N)
            heads.append(Hk)
        H_stack = torch.stack(heads, dim=0)  # (K,N,N)
        w = F.softmax(self.alpha, dim=0).view(self.K, 1, 1)
        H_fused = (H_stack * w).sum(0)       # (N,N)
        return H_stack, H_fused

    def _incidence_from_kernels(self, H_stack: torch.Tensor):
        """
        Make soft incidence B (N,E) by turning each row (sender i) of each head k into a hyperedge.
        If topk < N, keep only top-k targets per row and renormalize.
        """
        K, N, _ = H_stack.shape
        E = K * N
        device = H_stack.device
        B = torch.zeros(N, E, device=device)
        W = torch.ones(E, device=device)  # light wrapper: unit hyperedge weights

        col = 0
        if self.topk is not None and self.topk < N:
            for k in range(K):
                Hk = H_stack[k]  # (N,N)
                vals, idx = torch.topk(Hk, k=self.topk, dim=1)
                vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-9)
                for i in range(N):
                    B[idx[i], col + i] = vals[i]
                col += N
        else:
            for k in range(K):
                Hk = H_stack[k]
                # Column e=(k,i) is Hk[i,:]; put as a column in B (needs transpose)
                B[:, col:col+N] = Hk.t()
                col += N
        return B, W

    @staticmethod
    def _power_iteration_lambda_max(L: torch.Tensor, iters: int = 5):
        """Estimate largest eigenvalue of (symmetric) L via power iteration."""
        N = L.size(0)
        v = torch.randn(N, 1, device=L.device)
        v = v / (v.norm() + 1e-9)
        lam = None
        for _ in range(iters):
            v = L @ v
            nrm = v.norm() + 1e-9
            v = v / nrm
            lam = (v.t() @ (L @ v)).squeeze()  # Rayleigh quotient
        return lam.clamp(min=1e-3)

    def _hypergraph_laplacian(self, B: torch.Tensor, W: torch.Tensor):
        """
        L = I - Dv^{-1/2} B W De^{-1} B^T Dv^{-1/2}
        Optional Laplacian dropout: randomly drop columns (hyperedges) of B during training.
        """
        if self.training and self.lap_dropout_p > 0.0:
            E = B.size(1)
            # dropout on columns (hyperedges)
            keep = torch.rand(E, device=B.device) > self.lap_dropout_p
            B = B[:, keep]
            W = W[keep]

        N, E = B.shape
        # De (E,): hyperedge degree (sum over vertices)
        De = B.sum(dim=0) + 1e-9
        # Dv (N,): vertex degree (sum over hyperedges, weighted)
        Dv = (B * W.unsqueeze(0)).sum(dim=1) + 1e-9

        Dv_inv_sqrt = (1.0 / Dv).sqrt()
        De_inv = 1.0 / De

        # S = B W De^{-1} B^T
        scale = (W * De_inv).unsqueeze(0)  # (1,E)
        BWDe = B * scale                   # (N,E)
        S = BWDe @ B.t()                   # (N,N)

        # Normalize by Dv^{-1/2}
        S_norm = (Dv_inv_sqrt.unsqueeze(1) * S) * Dv_inv_sqrt.unsqueeze(0)
        L = torch.eye(N, device=B.device) - S_norm
        return L

    def _chebyshev_filter(self, L: torch.Tensor, X: torch.Tensor):
        """
        Chebyshev filter: sum_{t=0}^T diag(θ_t) T_t(L̃) X, where L̃=2L/lambda_max - I.
        Uses lambda_max from power iteration for stable scaling; adds ELU afterwards.
        """
        # Estimate λ_max
        lam_max = self._power_iteration_lambda_max(L.detach(), iters=5)
        self.last_lambda_max = lam_max.detach().clone()

        N = L.size(0)
        L_tilde = (2.0 / lam_max) * L - torch.eye(N, device=L.device)

        T0 = X                                            # (N,D)
        out = T0 * self.theta[0].view(1, -1)             # broadcast over features

        if self.cheb_T >= 1:
            T1 = L_tilde @ X
            out = out + T1 * self.theta[1].view(1, -1)

        for t in range(2, self.cheb_T + 1):
            T2 = 2.0 * (L_tilde @ T1) - T0
            out = out + T2 * self.theta[t].view(1, -1)
            T0, T1 = T1, T2

        # Nonlinearity after spectral mix
        return F.elu(out)

    # ---------- forward ----------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N, D_in) node features
        Returns: Z: (N, D_out)
        """
        N, D = X.shape

        # 1) Kernel stack and fused kernel
        H_stack, H_fused = self._build_kernel_stack(X)  # (K,N,N), (N,N)

        # 2) Optional sender/receiver gating
        if self.use_attention:
            g = self.att_mlp(X).squeeze(-1)            # (N,)
            H_fused = H_fused * g.view(N,1) * g.view(1,N)

        # 3) Incidence + Laplacian
        B, W = self._incidence_from_kernels(H_stack)   # (N,E), (E,)
        L = self._hypergraph_laplacian(B, W)           # (N,N)

        # 4) Spectral path (Chebyshev)
        Z_spec = self._chebyshev_filter(L, X)          # (N,D)

        # 5) Kernel diffusion path
        Z_kern = H_fused @ X                           # (N,D)

#        Z_spec_n = F.layer_norm(Z_spec, Z_spec.shape[-1:])
#        Z_kern_n = F.layer_norm(Z_kern, Z_kern.shape[-1:])

        # 6) Learnable blend
        rho = torch.sigmoid(self.rho_raw)              # (0,1)
        Z_mix = rho * Z_spec + (1.0 - rho) * Z_kern

        # 7) Projection to output dim
        Z = self.proj(Z_mix)
        # Save for interpretability
        self.H_kernels = H_stack.detach()
        self.H_fused   = H_fused.detach()
        self.B_inc     = B.detach()
        self.L_hg      = L.detach()
        self.rho_value = rho.detach()
        return Z



class CMSIPlus(nn.Module):
    # CMSIPlus: Cross-Modality Set Interaction Module
    def __init__(self, d, method='cross_attention', lowrank_r=16,
                 info_theory=False, use_ot=False, ot_eps=0.05, ot_iters=20):
        """
        Args:
            d (int): Embedding dimension per subject.
            method (str): 'cross_attention' or 'lowrank'.
            lowrank_r (int): Rank for low-rank factorization.
            info_theory (bool): Use KL-based Gaussian similarity.
            use_ot (bool): Use entropic OT (Sinkhorn) matching.
            ot_eps (float): Entropy regularization for OT.
            ot_iters (int): Sinkhorn iterations.
        """
        super().__init__()
        self.method = method
        self.info_theory = info_theory
        self.use_ot = use_ot
        self.ot_eps = ot_eps
        self.ot_iters = ot_iters

        if info_theory:
            # Gaussian parameters for each modality
            self.mu_m, self.logv_m = nn.Linear(d, d), nn.Linear(d, d)
            self.mu_n, self.logv_n = nn.Linear(d, d), nn.Linear(d, d)
            self.tau = nn.Parameter(torch.tensor(1.0))
        elif method == 'cross_attention':
            self.W_Q, self.W_K = nn.Linear(d, d), nn.Linear(d, d)
        elif method == 'lowrank':
            self.W1 = nn.Linear(d, lowrank_r, bias=False)
            self.W2 = nn.Linear(d, lowrank_r, bias=False)
            self.bias = nn.Parameter(torch.zeros(1))

    def _sinkhorn(self, cost):
        """Entropic OT via Sinkhorn iterations."""
        n = cost.size(0)
        mu = cost.new_full((n,), 1 / n)
        nu = mu.clone()
        Kmat = torch.exp(-cost / self.ot_eps)
        u = torch.ones_like(mu)
        for _ in range(self.ot_iters):
            u = mu / (Kmat @ (nu / (Kmat.t() @ u)))
        v = nu / (Kmat.t() @ u)
        return torch.outer(u, v) * Kmat  # transport plan T

    def forward(self, H_m, H_n):
        """
        Args:
            H_m, H_n: (N, d) modality embeddings.
        Returns:
            (N, N) interaction matrix.
        """
        if self.info_theory:
            # Gaussian KL divergence: KL(N(m1,v1)||N(m2,v2))
            mu_m, lv_m = self.mu_m(H_m), self.logv_m(H_m)
            mu_n, lv_n = self.mu_n(H_n), self.logv_n(H_n)
            v_m, v_n = lv_m.exp(), lv_n.exp()
            kl = 0.5 * ((v_m / v_n).sum(-1, True)
                        + ((mu_n - mu_m) ** 2 / v_n).sum(-1, True)
                        - H_m.size(-1)
                        + (lv_n - lv_m).sum(-1, True))
            # Convert divergence to similarity
            return torch.exp(-kl / self.tau)

        if self.use_ot:
            # squared-Euclid cost
            cost = torch.cdist(H_m, H_n, p=2).pow(2)
            return self._sinkhorn(cost)

        if self.method == 'cross_attention':
            H_m = F.layer_norm(H_m, H_m.shape[-1:])
            H_n = F.layer_norm(H_n, H_n.shape[-1:])
            Q, K = self.W_Q(H_m), self.W_K(H_n)
            scores = Q @ K.t() / math.sqrt(H_m.size(-1))
            return torch.softmax(scores, dim=-1)

        # low-rank bilinear
        U, V = self.W1(H_m), self.W2(H_n)
        return U @ V.t() + self.bias


class VWVGraph(nn.Module):
    # Subject graph via cognitive embedding, with optional Riemannian or Hypergraph kernels
    def __init__(self, d, K=4, use_mask=False, use_riemann=False, use_hypergraph=True):
        """
        Args:
            d (int): Cognitive feature dimension.
            K (int): Number of kernel heads.
            use_mask (bool): Learn edge gating.
            use_riemann (bool): Approximate SPD manifold distance.
            use_hypergraph (bool): Use cosine hypergraph kernel.
        """
        super().__init__()
        self.K = K
        self.use_mask = use_mask
        self.use_riemann = use_riemann
        self.use_hg = use_hypergraph
        self.W_k = nn.ParameterList([nn.Parameter(torch.randn(d, d)) for _ in range(K)])
        self.alpha = nn.Parameter(torch.ones(K))
        if use_mask:
            self.edge_mlp = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(),
                                          nn.Linear(d, 1), nn.Sigmoid())

    def forward(self, V):
        """
        Args:
            V: (N, d) cognitive vectors.
        Returns:
            H: (N, N) fused affinity matrix.
        """
        N = V.size(0)
        H_all = []
        for W in self.W_k:
            if self.use_hg:
                # hypergraph kernel = cosine similarity squared
                Vn = F.normalize(V, dim=1)
                H_k = (Vn @ Vn.t()).pow(2)
            elif self.use_riemann:
                # approximate SPD distance via outer-product flatten
                C = torch.bmm(V.unsqueeze(-1), V.unsqueeze(1))
                H_k = -torch.cdist(C.view(N, -1), C.view(N, -1))
            else:
                # standard bilinear head
                H_k = V @ W @ V.t()
                H_k = torch.softmax(H_k, dim=-1)

            if self.use_mask:
                vi = V.unsqueeze(1).expand(-1, N, -1)
                vj = V.unsqueeze(0).expand(N, -1, -1)
                mask = self.edge_mlp(torch.cat([vi, vj], -1)).squeeze(-1)
                H_k = H_k * mask

            H_all.append(H_k)

        H_stack = torch.stack(H_all, dim=0)  # (K,N,N)
        weights = torch.softmax(self.alpha, dim=0)  # head weights
        H = (H_stack * weights.view(-1, 1, 1)).sum(0)  # fuse
        # save for interpretability
        self.H_stack = H_stack.detach()
        self.alpha_weights = weights.detach()
        self.final_H = H.detach()
        return H




class HyperCoCoFusion(nn.Module):
    def __init__(self, encoder_fc, encoder_sc, encoder_morph,
                 cmsi_d, vwv_d, label_dim,
                 cmsi_method='cross_attention', info_theory=False,
                 vwv_riemann=False,
                 use_spectral_hypergraph=False,
                 use_residual=False):
        """
        Args:
            encoder_fc, encoder_sc, encoder_morph: modality encoders.
            cmsi_d (int): CMSI embedding dim.
            vwv_d (int): cognitive dim for VWV.
            label_dim (int): number of labels C.
            use_spectral_hypergraph (bool): use HypergraphFusion.
            use_residual (bool): apply residual skip.
        """
        super().__init__()
        self.encoder_fc = encoder_fc
        self.encoder_sc = encoder_sc
        self.encoder_morph = encoder_morph
        
        self.cmsi_fc_sc    = CMSIPlus(cmsi_d, method=cmsi_method, info_theory=info_theory) # fc → sc (values = sc)
        self.cmsi_sc_fc    = CMSIPlus(cmsi_d, method=cmsi_method, info_theory=info_theory) # sc → fc (values = fc)

        self.cmsi_fc_morph = CMSIPlus(cmsi_d, method=cmsi_method, info_theory=info_theory) # fc → morph (values = morph)
        self.cmsi_morph_fc = CMSIPlus(cmsi_d, method=cmsi_method, info_theory=info_theory) # morph → fc (values = fc)

        self.cmsi_sc_morph = CMSIPlus(cmsi_d, method=cmsi_method, info_theory=info_theory) # sc → morph (values = morph)
        self.cmsi_morph_sc = CMSIPlus(cmsi_d, method=cmsi_method, info_theory=info_theory) # morph → sc (values = sc)
        
        self.vwv = VWVGraph(vwv_d, use_riemann=vwv_riemann)
        self.gamma_raw = nn.Parameter(torch.tensor(0.0001))
        
        self.use_spectral_hypergraph = use_spectral_hypergraph
        self.use_residual = use_residual
        
        if use_spectral_hypergraph:#changed from three due to the things for fc
            self.hypergraph = HypergraphFusion(input_dim=cmsi_d * 6, output_dim=cmsi_d * 6)
        if use_residual:
            self.beta = nn.Parameter(torch.tensor(0.5))
        self.fusion_layer = nn.Linear(cmsi_d * 6, 128) #same reason here
        self.output_layer = nn.Sequential(nn.ReLU(), nn.Linear(128, label_dim))

    def _row_norm(self, H: torch.Tensor, eps: float = 1e-8):
        rowsum = H.sum(dim=1, keepdim=True)
        return H / (rowsum + eps)

    def forward(self, fc_input, sc_input, morph_input, cog_scores):
        # Encode each modality
        z_fc = self.encoder_fc(fc_input)  # (N, d)
        z_sc = self.encoder_sc(sc_input)  # (N, d)
        z_morph = self.encoder_morph(morph_input)  # (N, d)
        # Cross-modal CMSI interactions
        # Convention: source→target uses @ z_target (values = target)
        int_fc_sc    = self.cmsi_fc_sc(z_fc,z_sc)
        int_sc_fc    = self.cmsi_sc_fc(z_sc,z_fc)

        int_fc_morph = self.cmsi_fc_morph(z_fc,    z_morph)
        int_morph_fc = self.cmsi_morph_fc(z_morph, z_fc)

        int_sc_morph = self.cmsi_sc_morph(z_sc,    z_morph)
        int_morph_sc = self.cmsi_morph_sc(z_morph, z_sc)
        # Save heatmaps
        self.compat_fc_sc, self.compat_fc_morph, self.compat_sc_morph = (
            int_fc_sc.detach(), int_fc_morph.detach(), int_sc_morph.detach()
        )
        self.compat_sc_fc, self.compat_morph_fc, self.compat_morph_sc = int_sc_fc.detach(), int_morph_fc.detach(), int_morph_sc.detach()
        
        agg_fc_sc    = int_fc_sc    @ z_sc # fc → sc:    carries sc values (z_sc)
        agg_sc_fc    = int_sc_fc    @ z_fc # sc → fc:    carries fc values (z_fc)

        agg_fc_morph = int_fc_morph @ z_morph  # fc → morph: carries morph values (z_morph)
        agg_morph_fc = int_morph_fc @ z_fc # morph → fc: carries fc values (z_fc)

        agg_sc_morph = int_sc_morph @ z_morph # sc → morph: carries morph values (z_morph)
        agg_morph_sc = int_morph_sc @ z_sc  # morph → sc: carries sc values (z_sc)
        

        interaction_repr = torch.cat([agg_fc_sc, agg_fc_morph, agg_sc_morph, agg_sc_fc, agg_morph_fc,agg_morph_sc], dim=-1)

        # Fusion variants
        if self.use_spectral_hypergraph:
            #subject graph
            Z_temp = self.hypergraph(interaction_repr)
            H_vwv = self.vwv(cog_scores)               # (N,N) subject–subject
            H_vwv = self._row_norm(H_vwv)
            Z_fused = H_vwv @ Z_temp #propogate accross similar subjects
            
        else:
            H_vwv = self.vwv(cog_scores)
            Z_temp = H_vwv @ interaction_repr

        if self.use_residual:
            if self.use_spectral_hypergraph:
                Z_fused = self.beta * Z_fused + (1 - self.beta) * interaction_repr
            else:
                Z_fused = self.beta * Z_temp + (1 - self.beta) * interaction_repr

        # Prediction
        # at this point we are doing regression
#        logits = self.output_layer(self.fusion_layer(Z_fused))
#        probs = torch.sigmoid(logits)
        probs = self.output_layer(self.fusion_layer(Z_fused))

        # Comorbidity matrix for interpretability
        self.comorb_matrix = probs.t() @ probs
        return probs

    def get_interpretability(self):
        return {
            'compat_fc_sc':     self.compat_fc_sc,
            'compat_fc_morph':  self.compat_fc_morph,
            'compat_sc_morph':  self.compat_sc_morph,
            'compat_sc_fc':     self.compat_sc_fc,      # NEW
            'compat_morph_fc':  self.compat_morph_fc,   # NEW
            'compat_morph_sc':  self.compat_morph_sc,   # NEW
            'vwv_heads': self.vwv.H_stack,
            'vwv_alpha': self.vwv.alpha_weights,
            'vwv_final': self.vwv.final_H,
            'comorb_matrix': self.comorb_matrix
        }


#if __name__ == '__main__':
#    from encoders import DummyEncoder, GATEncoder
#
#    # ---- simulate a tiny batch -----------------------------------------
#    N, D, d, C = 16, 400, 10, 6  # subjects, ROI, cog-dim, labels
#    fc = torch.randn(N, D, D)  # functional connectivity
#    sc = torch.randn(N, D, D)  # structural connectivity
#    morph = torch.randn(N, D, 12)  # morphometric features
#    cog = torch.randn(N, d)  # cognitive scores
#    labels = torch.randint(0, 2, (N, C)).float()  # multi-label targets
#
#    # ---- build the model with fusion variants enabled ------------------
#    cmsi_dim = 32
#    model = HyperCoCoFusion(
#        encoder_fc=DummyEncoder(D, cmsi_dim),
#        encoder_sc=DummyEncoder(D, cmsi_dim),
#        encoder_morph=DummyEncoder(12, cmsi_dim),
#        cmsi_d=cmsi_dim,
#        vwv_d=d,
#        label_dim=C,
#        cmsi_method='cross_attention',
#        info_theory=False,
#        vwv_riemann=False,
#        use_spectral_hypergraph=True,
#        use_residual=True
#    )
#
#    # ---- forward + backward test --------------------------------------
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#    model.to(device)
#    fc, sc, morph, cog, labels = [t.to(device) for t in (fc, sc, morph, cog, labels)]
#
#    criterion = nn.BCELoss()
#    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#    model.train()
#    probs = model(fc, sc, morph, cog)  # forward
#    task_loss = criterion(probs, labels)
#    task_loss.backward()  # backward
#    optimizer.step()
#
#    print(f"✅ Fusion variant test passed — Loss: {task_loss.item():.4f}")
