"""
Gated Progressive Representation Module (GPRM) - Paper Section 3.3

Funnel-shaped progressive feature compression with gated feature recalibration.
h_t = max(floor(h_{t-1}/2), h_min), where h_min = 64.
Gate: z_tilde = z_T * g, g = sigma(BN(W_g * z_T))  (Eq.7)

Output dimension: FEAT_DIM (deep feature f_deep).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM = 10

H_MIN = 64  # minimum hidden layer width (paper Section 3.3)


class _ProgressiveBNN(nn.Module):
    """
    Funnel-shaped progressive compression + gated recalibration.

    Architecture:
    - Progressive layers: each halves the dimension until h_min
    - Gated recalibration: g = sigma(BN(W_g * z_T))  (Eq.7)
    - Final projection: z_T -> FEAT_DIM
    """
    def __init__(self, d_in: int, n_cls: int, dropout: float = 0.2):
        super().__init__()

        self.hidden_dims = self._compute_hidden_dims(d_in)

        # Funnel-shaped progressive compression submodule
        self.progressive_layers = nn.ModuleList()
        prev_dim = d_in
        for h_dim in self.hidden_dims:
            self.progressive_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim, bias=False),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = h_dim

        # Gated feature recalibration submodule (Eq.7)
        final_dim = self.hidden_dims[-1] if len(self.hidden_dims) > 0 else d_in
        self.gate_layer = nn.Sequential(
            nn.Linear(final_dim, final_dim, bias=False),
            nn.BatchNorm1d(final_dim),
            nn.Sigmoid()
        )

        # Projection to output feature dimension
        self.proj = nn.Linear(final_dim, FEAT_DIM)
        self.head = nn.Linear(FEAT_DIM, n_cls)

    def _compute_hidden_dims(self, d_in: int):
        """h_t = max(floor(h_{t-1}/2), h_min)"""
        dims = []
        current = d_in
        while current > H_MIN:
            current = max(current // 2, H_MIN)
            dims.append(current)
        return dims

    def forward(self, x: torch.Tensor):
        # Progressive compression
        h = x
        for layer in self.progressive_layers:
            h = layer(h)

        # Gated recalibration: z_tilde = z_T * g (Eq.7)
        gate = self.gate_layer(h)
        h_gated = h * gate

        # Project to output dimension
        feat = self.proj(h_gated)
        logits = self.head(feat)
        return feat, logits


class ProgressiveBNNExtractor(nn.Module):
    """
    GPRM feature extractor (public interface).

    Usage:
        ext = ProgressiveBNNExtractor(d_in, n_cls).to(device)
        F_tr = ext.fit_transform(X_tr, y_tr)
        F_te = ext.transform(X_te)
    """
    def __init__(self, d_in: int, n_cls: int, dropout: float = 0.2):
        super().__init__()
        self.net = _ProgressiveBNN(d_in, n_cls, dropout)

    @torch.no_grad()
    def transform(self, X: torch.Tensor, batch_size: int = 512):
        self.eval()
        outs = []
        for i in range(0, len(X), batch_size):
            f, _ = self.net(X[i:i+batch_size])
            outs.append(f)
        return torch.cat(outs, dim=0)

    def fit_transform(
        self, X: torch.Tensor, y: torch.Tensor,
        epochs: int = 40, batch_size: int = 64,
        lr: float = 1e-3, verbose: bool = False,
        sample_weights: torch.Tensor = None
    ):
        self.fit(X, y, epochs, batch_size, lr, verbose, sample_weights)
        with torch.no_grad():
            return self.transform(X, batch_size)

    def fit(
        self, X: torch.Tensor, y: torch.Tensor,
        epochs: int = 40, batch_size: int = 64,
        lr: float = 1e-3, verbose: bool = False,
        sample_weights: torch.Tensor = None
    ):
        device = next(self.parameters()).device
        X, y = X.detach().to(device).float(), y.to(device)

        if sample_weights is not None:
            sample_weights = sample_weights.detach().to(device).float()
            sample_weights = sample_weights / sample_weights.mean()

        if sample_weights is not None:
            dataset = torch.utils.data.TensorDataset(X, y, sample_weights)
        else:
            dataset = torch.utils.data.TensorDataset(X, y)

        g = torch.Generator()
        g.manual_seed(0)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=g
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            self.train()
            tot = 0.0
            for batch in loader:
                if sample_weights is not None:
                    xb, yb, wb = batch
                else:
                    xb, yb = batch
                    wb = None

                opt.zero_grad()
                _, logits = self.net(xb)

                if wb is not None:
                    loss = F.cross_entropy(logits, yb, reduction='none')
                    loss = (loss * wb).mean()
                else:
                    loss = F.cross_entropy(logits, yb)

                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)

            if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
                print(f"[GPRM] epoch {ep:03d}/{epochs} | CE={tot/len(X):.4f}")

        self.eval()
