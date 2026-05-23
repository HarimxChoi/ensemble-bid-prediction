# -*- coding: utf-8 -*-
"""
src/models/r2ccp_2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class R2CCPConfig:
    """R2CCP training configuration."""
    model_path: str = 'model.pth'
    max_epochs: int = 100
    range_size: int = 50
    alpha: float = 0.1
    
    # Architecture
    ffn_hidden_dim: int = 128
    ffn_num_layers: int = 3
    ffn_activation: str = 'relu'
    dropout_prob: float = 0.0
    
    # Training
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-4
    
    # Calibration
    cal_size: float = 0.2
    
    # Loss weights
    entropy_weight: float = 3.0
    loss_weight: float = 5.0
    
    # Adaptive range
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    
    # Misc
    seed: int = 42
    enable_checkpointing: bool = False


# =============================================================================
# MLP MODEL
# =============================================================================

class MLPModel(nn.Module):
    """Simple MLP for softmax probability prediction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        act_fn = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# =============================================================================
# LIGHTNING MODULE
# =============================================================================

class R2CCPModule(pl.LightningModule):
    """
    PyTorch Lightning module for R2CCP training.
    
    Loss: Cross-entropy + entropy regularization
    - Cross-entropy: encourages correct bin prediction
    - Entropy regularization: prevents collapse to one-hot
    """
    
    def __init__(
        self,
        input_dim: int,
        range_vals: torch.Tensor,
        config: R2CCPConfig
    ):
        super().__init__()
        
        self.range_vals = range_vals
        self.n_bins = len(range_vals)
        self.config = config
        
        self.model = MLPModel(
            input_dim=input_dim,
            output_dim=self.n_bins,
            hidden_dim=config.ffn_hidden_dim,
            num_layers=config.ffn_num_layers,
            activation=config.ffn_activation,
            dropout=config.dropout_prob
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        # Compute step size for index calculation
        self.step_val = (range_vals[-1] - range_vals[0]) / (len(range_vals) - 1)
        self.min_val = range_vals[0]
        
        self.save_hyperparameters(ignore=['range_vals'])
    
    def forward(self, x):
        """Forward pass: returns logits."""
        return self.model(x)
    
    def get_softmax(self, x):
        """Get softmax probabilities."""
        return self.softmax(self.forward(x))
    
    def _compute_target_indices(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute interpolated target indices for soft cross-entropy.
        
        Returns:
            indices_down: Floor indices
            indices_up: Ceil indices  
            weights_up: Interpolation weight for upper index
        """
        # Compute fractional index
        idx_float = (y.squeeze() - self.min_val) / self.step_val
        
        indices_down = torch.floor(idx_float).long()
        indices_up = torch.ceil(idx_float).long()
        
        # Clamp to valid range
        indices_down = torch.clamp(indices_down, 0, self.n_bins - 1)
        indices_up = torch.clamp(indices_up, 0, self.n_bins - 1)
        
        # Interpolation weights
        weights_up = idx_float - torch.floor(idx_float)
        weights_down = 1 - weights_up
        
        return indices_down, indices_up, weights_up
    
    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        
        probs = self.softmax(logits)
        
        indices_down, indices_up, weights_up = self._compute_target_indices(y)
        weights_down = 1 - weights_up
        
        prob_down = probs.gather(1, indices_down.view(-1, 1)).squeeze(1)
        prob_up = probs.gather(1, indices_up.view(-1, 1)).squeeze(1)
        
        prob_at_y = prob_down * weights_down + prob_up * weights_up
        
        ce_loss = -torch.log(prob_at_y + 1e-10).mean()
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
        
        loss = self.config.loss_weight * ce_loss - self.config.entropy_weight * entropy
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self._compute_loss(logits, y)
        
        # Check for NaN
        if torch.isnan(loss):
            # Fallback to simple CE
            probs = self.softmax(logits)
            indices_down, _, _ = self._compute_target_indices(y)
            loss = F.cross_entropy(logits, indices_down)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self._compute_loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_epochs
        )
        return [optimizer], [scheduler]


# =============================================================================
# CONFORMAL PREDICTION
# =============================================================================

def compute_conformity_scores(
    range_vals: torch.Tensor,
    y_cal: torch.Tensor,
    softmax_probs: torch.Tensor
) -> torch.Tensor:
    
    n_bins = len(range_vals)
    step_val = (range_vals[-1] - range_vals[0]) / (n_bins - 1)
    min_val = range_vals[0]
    max_val = range_vals[-1]
    
    y_squeezed = y_cal.squeeze()
    
    # Compute fractional indices
    idx_float = (y_squeezed - min_val) / step_val
    indices_down = torch.floor(idx_float).long()
    indices_up = torch.ceil(idx_float).long()
    
    # Clamp
    indices_down = torch.clamp(indices_down, 0, n_bins - 1)
    indices_up = torch.clamp(indices_up, 0, n_bins - 1)
    
    # Interpolation weights
    weights_up = idx_float - torch.floor(idx_float)
    weights_down = 1 - weights_up
    
    # Handle out-of-range
    out_of_range = (y_squeezed < min_val) | (y_squeezed > max_val)
    
    # Gather and interpolate
    prob_down = softmax_probs.gather(1, indices_down.view(-1, 1)).squeeze(1)
    prob_up = softmax_probs.gather(1, indices_up.view(-1, 1)).squeeze(1)
    
    scores = prob_down * weights_down + prob_up * weights_up
    scores[out_of_range] = 0.0
    
    return scores


def compute_conformal_threshold(
    range_vals: torch.Tensor,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    model: nn.Module,
    alpha: float
) -> float:

    model.eval()
    with torch.no_grad():
        if isinstance(X_cal, np.ndarray):
            X_cal = torch.tensor(X_cal, dtype=torch.float32)
        if isinstance(y_cal, np.ndarray):
            y_cal = torch.tensor(y_cal, dtype=torch.float32)
        
        logits = model(X_cal)
        softmax_probs = F.softmax(logits, dim=1)
        
        scores = compute_conformity_scores(range_vals, y_cal, softmax_probs)
        
        # Filter valid scores
        valid_scores = scores[scores > 0]
        if len(valid_scores) == 0:
            return 0.0
        
        # Alpha quantile (matches pip R2CCP)
        threshold = torch.quantile(valid_scores, alpha).item()
    
    return threshold


def find_intervals_above_threshold(
    bin_centers: np.ndarray,
    softmax_probs: np.ndarray,
    threshold: float
) -> List[Tuple[float, float]]:
    """
    Find all contiguous intervals where softmax >= threshold.
    """
    intervals = []
    start = None
    
    n_bins = len(bin_centers)
    if n_bins < 2:
        return intervals
    
    # Bin width for edge handling
    bin_width = (bin_centers[-1] - bin_centers[0]) / (n_bins - 1)
    
    # Check if first bin is above threshold
    if softmax_probs[0] >= threshold:
        start = bin_centers[0] - bin_width / 2
    
    for i in range(n_bins - 1):
        x1, x2 = bin_centers[i], bin_centers[i + 1]
        y1, y2 = softmax_probs[i], softmax_probs[i + 1]
        
        # Check for threshold crossing
        if min(y1, y2) <= threshold < max(y1, y2):
            # Linear interpolation for crossing point
            if abs(y2 - y1) > 1e-10:
                x_cross = x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
            else:
                x_cross = (x1 + x2) / 2
            
            if start is None:
                # Rising edge: start interval
                start = x_cross
            else:
                # Falling edge: close interval
                intervals.append((float(start), float(x_cross)))
                start = None
    
    # Close final interval if still open
    if start is not None:
        intervals.append((float(start), float(bin_centers[-1] + bin_width / 2)))
    
    return intervals


def get_prediction_intervals(
    X: np.ndarray,
    model: nn.Module,
    range_vals: torch.Tensor,
    threshold: float,
    scaler_y: StandardScaler
) -> List[List[Tuple[float, float]]]:
    model.eval()
    
    # Convert range_vals to original space
    range_vals_np = range_vals.cpu().numpy() if isinstance(range_vals, torch.Tensor) else range_vals
    bin_centers_orig = scaler_y.inverse_transform(range_vals_np.reshape(-1, 1)).flatten()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        softmax_probs = F.softmax(logits, dim=1).cpu().numpy()
    
    all_intervals = []
    
    for i in range(len(X)):
        intervals = find_intervals_above_threshold(
            bin_centers_orig,
            softmax_probs[i],
            threshold
        )
        
        # Fallback if no intervals found
        if len(intervals) == 0:
            intervals = [(float(bin_centers_orig[0]), float(bin_centers_orig[-1]))]
        
        all_intervals.append(intervals)
    
    return all_intervals


# =============================================================================
# MAIN R2CCP CLASS
# =============================================================================

class R2CCP:
    """
    Unified R2CCP
    """
    
    def __init__(self, options: Dict):
        self.config = R2CCPConfig()
        
        # Apply options
        for key, value in options.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.range_vals = None
        self.bin_edges = None
        self.conformal_threshold = None
        self.train_X = None
        self.train_y = None
        self.q_value = None  # For API compatibility
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train R2CCP model.
        """
        # Seed
        pl.seed_everything(self.config.seed)
        
        # Reshape y if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Scale features
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Scale target
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Store for later
        self.train_X = X_scaled
        self.train_y = y_scaled
        
        # Compute range in SCALED space
        if self.config.y_min is not None and self.config.y_max is not None:
            # Adaptive range provided (in original space) - convert to scaled
            y_min_scaled = (self.config.y_min - self.scaler_y.mean_[0]) / self.scaler_y.scale_[0]
            y_max_scaled = (self.config.y_max - self.scaler_y.mean_[0]) / self.scaler_y.scale_[0]
        else:
            # Auto range from data
            y_min_scaled = y_scaled.min() - 0.5
            y_max_scaled = y_scaled.max() + 0.5
        
        self.range_vals = torch.linspace(
            float(y_min_scaled),
            float(y_max_scaled),
            self.config.range_size
        )
        
        # Store bin_edges in ORIGINAL space for API compatibility
        range_vals_np = self.range_vals.numpy()
        self.bin_edges = self.scaler_y.inverse_transform(range_vals_np.reshape(-1, 1)).flatten()
        
        # Train/cal split
        n = len(X_scaled)
        n_cal = max(int(n * self.config.cal_size), 10)
        n_train = n - n_cal
        
        X_train, X_cal = X_scaled[:n_train], X_scaled[n_train:]
        y_train, y_cal = y_scaled[:n_train], y_scaled[n_train:]
        
        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        cal_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_cal, dtype=torch.float32),
            torch.tensor(y_cal, dtype=torch.float32)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(X_train)),
            shuffle=True
        )
        cal_loader = torch.utils.data.DataLoader(
            cal_dataset,
            batch_size=len(X_cal),
            shuffle=False
        )
        
        # Create model
        input_dim = X_train.shape[1]
        module = R2CCPModule(input_dim, self.range_vals, self.config)
        
        # Check for existing model
        import os
        if os.path.exists(self.config.model_path):
            module.model.load_state_dict(torch.load(self.config.model_path))
        else:
            # Train
            trainer_kwargs = {
                'max_epochs': self.config.max_epochs,
                'enable_progress_bar': False,
                'enable_checkpointing': self.config.enable_checkpointing,
                'logger': False,
                'gradient_clip_val': 5.0,  # Stability
            }
            
            if torch.cuda.is_available():
                trainer_kwargs['accelerator'] = 'gpu'
                trainer_kwargs['devices'] = [0]
            else:
                trainer_kwargs['accelerator'] = 'cpu'
            
            trainer = pl.Trainer(**trainer_kwargs)
            trainer.fit(module, train_loader, cal_loader)
            
            # Save
            os.makedirs(os.path.dirname(self.config.model_path) or '.', exist_ok=True)
            torch.save(module.model.state_dict(), self.config.model_path)
        
        module.eval()
        self.model = module.model
        
        # Compute conformal threshold
        self.conformal_threshold = compute_conformal_threshold(
            self.range_vals,
            torch.tensor(X_cal, dtype=torch.float32),
            torch.tensor(y_cal, dtype=torch.float32),
            self.model,
            self.config.alpha
        )
        
        # Store q_value for API compatibility
        self.q_value = self.conformal_threshold
    
    def get_intervals(self, X: np.ndarray) -> List[List[Tuple[float, float]]]:
        """
        Get prediction intervals for new data.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler_X.transform(X)
        
        return get_prediction_intervals(
            X_scaled,
            self.model,
            self.range_vals,
            self.conformal_threshold,
            self.scaler_y
        )
    
    def get_coverage_length(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        intervals = self.get_intervals(X)
        
        y_flat = y.flatten()
        
        coverages = []
        lengths = []
        
        for i, interval_list in enumerate(intervals):
            y_val = y_flat[i]
            
            # Total length
            total_length = sum(hi - lo for lo, hi in interval_list)
            lengths.append(total_length)
            
            # Coverage check
            covered = any(lo <= y_val <= hi for lo, hi in interval_list)
            coverages.append(1.0 if covered else 0.0)
        
        return coverages, lengths
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler_X.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            best_indices = torch.argmax(probs, dim=1)
            
            # Get bin centers
            range_np = self.range_vals.numpy()
            best_scaled = range_np[best_indices.numpy()]
            
            # Convert to original space
            best_orig = self.scaler_y.inverse_transform(best_scaled.reshape(-1, 1))
        
        return best_orig


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def seed_everything(seed: int):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    X_train = np.random.rand(100, 5).astype(np.float32)
    y_train = (X_train[:, 0] * 0.05 + 0.98 + np.random.randn(100) * 0.01).astype(np.float32)
    
    X_test = np.random.rand(20, 5).astype(np.float32)
    y_test = (X_test[:, 0] * 0.05 + 0.98 + np.random.randn(20) * 0.01).astype(np.float32)
    
    model = R2CCP({
        'model_path': '/tmp/test_r2ccp.pth',
        'max_epochs': 20,
        'alpha': 0.1,
        'entropy_weight': 3.0,
        'loss_weight': 5.0,
    })
    
    model.fit(X_train, y_train)
    
    intervals = model.get_intervals(X_test)
    coverages, lengths = model.get_coverage_length(X_test, y_test.reshape(-1, 1))
    
    print(f"Coverage: {np.mean(coverages):.1%}")
    print(f"Avg Length: {np.mean(lengths):.4f}")
    print(f"Threshold: {model.conformal_threshold:.4f}")
    print(f"Sample intervals: {intervals[:3]}")