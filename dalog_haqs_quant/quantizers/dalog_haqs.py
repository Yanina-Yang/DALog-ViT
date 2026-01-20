import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from quantizers._ste import *
from quantizers.uniform import UniformQuantizer
from quantizers.logarithm import Log2Quantizer


class ImportanceMethod(Enum):
    """Importance calculation method enumeration"""
    GRADIENT_FIM = "gradient_fim"      # FIM diagonal elements based on gradients
    VARIANCE = "variance"              # Local variance based (default)
    MAGNITUDE = "magnitude"            # Activation magnitude based

class DALogQuantizer(Log2Quantizer):
    """
    DALog quantizer implementation with dynamic base optimization
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, 
                 ):
        super().__init__(n_bits, symmetric, channel_wise)
        
        self.r = 37.0  # DALog quantization base parameter
        self.training_mode = False
        
        self.min_r = 20.0  
        self.max_r = 50.0  
        self.adjust_factor = 0.1  # Adjustment factor
        
        self.register_buffer('q', torch.tensor([int(self.r)]))
        self.register_buffer('table1', torch.zeros((self.n_levels * 2)))
        self.register_buffer('table2', torch.zeros((self.n_levels * 2)))
        self.update_table()

    def init_from(self, x, *args, **kwargs):
        """Initialize DALog quantizer parameters from input data"""
        if self.inited:
            return
            
        # Calculate scale parameter (inherited from Log2Quantizer logic)
        if self.channel_wise:
            # Calculate scale per channel
            if x.dim() == 4:  # Conv layer [N, C, H, W]
                scale = x.abs().amax(dim=(0, 2, 3), keepdim=True)
            elif x.dim() == 2:  # Linear layer [N, C]
                scale = x.abs().amax(dim=0, keepdim=True)
            else:
                scale = x.abs().max()
        else:
            scale = x.abs().max()
            
        self.register_buffer('scale', scale)
        self.inited = True

    def update_table(self):
        """Update DALog quantization table"""
        for i in range(0, self.n_levels * 2):
            q_val = float(self.q.item())
            r_val = float(self.r)
            
            # Calculate using Python scalars
            exponent = -((q_val * i) % r_val) / r_val
            val = (2 ** exponent) * (4 * self.n_levels - 2) / (4 * self.n_levels - 2)
            
            # Use math.floor to ensure result is Python scalar
            table1_val = math.floor(i * q_val / r_val)
            
            self.table1[i].data.copy_(torch.tensor(table1_val))
            self.table2[i].data.copy_(torch.tensor(val))

    def adaptive_base(self, x):
        """
        Dynamically search for optimal quantization base for different layer activation ranges
        """
 
        # Analyze frequency distribution of activation values
        x_flat = x.flatten()
        x_nonzero = x_flat[x_flat > 1e-15]
        
        if x_nonzero.numel() > 0:
            # Calculate distribution characteristics in log domain
            log_vals = -torch.log2(x_nonzero.detach() / self.scale.detach())
            
            # Dynamically adjust quantization base based on distribution density
            log_mean = log_vals.mean().detach()
            log_std = log_vals.std().detach()
            
            # Adaptively adjust r parameter to maintain frequency characteristics - using config parameters
            optimal_r = max(self.min_r, min(self.max_r, 
                                               self.r * (1 + self.adjust_factor * log_std)))
            if abs(optimal_r - self.r) > 1.0:
                self.r = optimal_r
                self.q.data.copy_(torch.tensor([int(self.r)])) 
                self.update_table()

    def forward(self, x):
        if self.n_bits == 32:
            return x
        
        if not self.inited:
            self.init_from(x)
        
        assert self.inited, "DALog quantizer must be initialized first"
        
        # Dynamic base optimization
        if self.training_mode:
            self.adaptive_base(x)
        
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0)
        
        # Select quantization method based on training mode
        if self.training_mode:
            x_quant = round_ste(-scaled_x.log2() * self.r / self.q)
            mask = (x_quant < 2 * self.n_levels)
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = (2 ** (-1 * x_quant * self.q / self.r) * self.scale)
        else:
            x_quant = torch.round(-scaled_x.log2() * self.r / self.q)
            mask = (x_quant < 2 * self.n_levels)
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = ((2 ** (-self.table1[x_quant.long()])) * \
                       self.table2[x_quant.long()] * self.scale)
        
        x_dequant = x_dequant * mask
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, ' \
               f'channel_wise={self.channel_wise}, q={self.q.item()})'

class HAQSQuantizer(nn.Module):
    """
    Hybrid quantizer for GELU activation values
    Integrates DALog logarithmic quantization advantages to achieve distribution-adaptive differentiated quantization
    """
    
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, 
                 importance_threshold: float = 0.5,
                 importance_method = "magnitude", 
                 fast_mode: bool = False,
                 neg_clip_bound: float = -0.17,
                 pos_energy_threshold: float = 0.99,
                 enable_preprocessing: bool = True):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.sym = symmetric
        self.drop_prob = 1.0
        self.inited = False
        self.training_mode = False
        
        # GELU distribution characteristic parameters - passed from config file
        self.neg_clip_bound = neg_clip_bound
        self.pos_energy_threshold = pos_energy_threshold
        self.enable_preprocessing = enable_preprocessing  # Control whether to perform activation preprocessing
        
        if isinstance(importance_method, str):
            method_map = {
                "magnitude": ImportanceMethod.MAGNITUDE,
                "variance": ImportanceMethod.VARIANCE, 
                "gradient_fim": ImportanceMethod.GRADIENT_FIM,
                "fim": ImportanceMethod.GRADIENT_FIM,  # Alias support
            }
            self.importance_method = method_map.get(importance_method.lower(), ImportanceMethod.MAGNITUDE)
        else:
            self.importance_method = importance_method
        
        self.importance_threshold = importance_threshold
        self.fast_mode = fast_mode
        
        # Sub-quantizers - dalog_r parameter is hardcoded
        self.dalog_quantizer = DALogQuantizer(n_bits, symmetric, channel_wise)
        self.uniform_quantizer = UniformQuantizer(n_bits, symmetric, channel_wise)
        
        # Initialize corresponding buffers based on importance method
        self._init_importance_buffers()
        
    def _init_importance_buffers(self):
        """Initialize corresponding buffers based on importance calculation method"""
        if self.importance_method == ImportanceMethod.GRADIENT_FIM:
            # FIM method requires historical sample buffer
            self.register_buffer('fim_scores', None)
            self.register_buffer('importance_mask', None)
            # FIM threshold buffer for inference
            self.register_buffer('cached_fim_threshold', None)
        else:
            # Other methods don't need buffers
            pass
    
    def init_from(self, x, *args, **kwargs):
        """Initialize quantizer parameters"""
        if self.inited:
            return
            
        # Calculate unified scale parameter
        if self.channel_wise:
            if x.dim() == 4:
                scale = x.abs().amax(dim=(0, 2, 3), keepdim=True)
            elif x.dim() == 2:
                scale = x.abs().amax(dim=0, keepdim=True)
            else:
                scale = x.abs().max()
        else:
            scale = x.abs().max()
        
        # Initialize DALog quantizer
        self.dalog_quantizer.scale = nn.Parameter(scale.clone())
        self.dalog_quantizer.inited = True
        
        # Initialize uniform quantizer
        if self.sym:
            uniform_scale = scale / (self.n_levels - 0.5)
            self.uniform_quantizer.scale = nn.Parameter(uniform_scale.clone())
        else:
            uniform_scale = scale / (2 * self.n_levels - 1)
            zero_point = torch.zeros_like(scale)
            self.uniform_quantizer.scale = nn.Parameter(uniform_scale.clone())
            self.uniform_quantizer.zero_point = nn.Parameter(zero_point.clone())
            
        self.uniform_quantizer.drop_prob = self.drop_prob
        self.uniform_quantizer.inited = True
        self.inited = True

    def init_training(self):
        """Initialize training mode"""
        self.training_mode = True
        self.dalog_quantizer.training_mode = True
        if hasattr(self.dalog_quantizer, 'init_training'):
            self.dalog_quantizer.init_training()
        if hasattr(self.uniform_quantizer, 'init_training'):
            self.uniform_quantizer.init_training()

    def end_training(self):
        """End training mode and clean up buffers"""
        self.training_mode = False
        self.dalog_quantizer.training_mode = False
        
        # Cache statistical information needed for inference for gradient_fim method
        if self.importance_method == ImportanceMethod.GRADIENT_FIM:
            if self.fim_scores is not None and self.fim_scores.numel() > 0:
                # Cache FIM threshold for inference use
                try:
                    self.cached_fim_threshold = torch.quantile(
                        self.fim_scores, 1.0 - self.importance_threshold
                    ).detach().cpu()
                   
                except:
                    mean_fim = self.fim_scores.mean()
                    std_fim = self.fim_scores.std()
                    self.cached_fim_threshold = (mean_fim + 1.0 * std_fim).detach().cpu()

            else:
                self.cached_fim_threshold = None
            
            # Clean up training buffers but keep inference buffers
            self.fim_scores = None
            self.importance_mask = None
        
        if hasattr(self.dalog_quantizer, 'end_training'):
            self.dalog_quantizer.end_training()
        if hasattr(self.uniform_quantizer, 'end_training'):
            self.uniform_quantizer.end_training()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear_fim_cache(self):
        """
        Dedicated method to clear FIM cache, called by block_recon.py
        """
        if self.importance_method == ImportanceMethod.GRADIENT_FIM:
            self.fim_scores = None
            self.importance_mask = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    

    def gelu_adaptive_clipping(self, x):
        """Adaptive clipping for GELU activation values"""
        x_neg = torch.clamp(x, min=self.neg_clip_bound, max=0.0)
        
        x_pos = torch.clamp(x, min=0.0)
        if x_pos.numel() > 0:
            x_pos_flat = x_pos.flatten()
            x_pos_nonzero = x_pos_flat[x_pos_flat > 0]
            if x_pos_nonzero.numel() > 0:
                energy = x_pos_nonzero.pow(2)
                sorted_energy, _ = torch.sort(energy)
                cumsum_energy = torch.cumsum(sorted_energy, dim=0)
                total_energy = cumsum_energy[-1]
                threshold_idx = torch.searchsorted(cumsum_energy, 
                                                 self.pos_energy_threshold * total_energy)
                if threshold_idx < len(sorted_energy):
                    energy_threshold = sorted_energy[threshold_idx]
                    pos_clip_bound = torch.sqrt(energy_threshold)
                    x_pos = torch.clamp(x_pos, max=pos_clip_bound)
        
        x_clipped = torch.where(x >= 0, x_pos, x_neg)
        return x_clipped

    def compute_importance(self, x, grad_output=None):
        """
        Unified importance calculation entry point
        Select corresponding calculation method based on importance_method
        """
        # Special case handling: when importance_threshold is 0, return all-False mask, indicating all use uniform quantization
        if self.importance_threshold == 0.0:
            return torch.zeros_like(x, dtype=torch.bool)
            
        if self.importance_method == ImportanceMethod.GRADIENT_FIM:
            # [Strict mode] GRADIENT_FIM method does not allow degradation, must use FIM calculation
            return self._compute_gradient_fim_importance(x, grad_output)
        elif self.importance_method == ImportanceMethod.VARIANCE:
            return self._compute_variance_importance(x)
        elif self.importance_method == ImportanceMethod.MAGNITUDE:
            return self._compute_magnitude_importance(x)
        else:
            raise ValueError(f"Unsupported importance calculation method: {self.importance_method}")

    def _compute_gradient_fim_importance(self, x, grad_output=None):
        """Method 1: Gradient-based FIM diagonal element importance calculation"""
        if grad_output is None:
            if not self.training and not self.training_mode:
                # Inference mode: use FIM statistics cached during training
                if hasattr(self, 'cached_fim_threshold') and self.cached_fim_threshold is not None:
                    # Use cached threshold for magnitude calculation
                    magnitude_flat = x.abs().detach().flatten()
                    # Use threshold ratio learned during training
                    importance_mask = (magnitude_flat > self.cached_fim_threshold).detach()
                    return importance_mask.reshape(x.shape)
            else:
                raise ValueError("Gradient FIM method requires grad_output parameter")
            
        # Calculate FIM diagonal elements
        fim_diag = (grad_output.abs().detach() * x.abs().detach())
        fim_flat = fim_diag.flatten().detach()
        
        # Fast mode uses statistical threshold
        if self.fast_mode:
            return self._compute_statistical_threshold_mask(fim_flat, x.shape)
        
        # Inference stage directly calculates threshold
        if not self.training and not self.training_mode:
            threshold = torch.quantile(fim_flat.detach(), 1.0 - self.importance_threshold).detach()
            importance_mask = (fim_flat.detach() > threshold).detach()
            return importance_mask.reshape(x.shape)
        
        # Training stage FIM management
        max_fim_samples = 5000
        
        if self.fim_scores is None:
            sample_size = min(1000, fim_flat.numel())
            indices = torch.randperm(fim_flat.numel())[:sample_size]
            self.fim_scores = fim_flat[indices].detach().cpu()
        else:
            sample_size = min(1000, fim_flat.numel())
            indices = torch.randperm(fim_flat.numel())[:sample_size]
            new_samples = fim_flat[indices].detach().cpu()
            
            if self.fim_scores.numel() + new_samples.numel() > max_fim_samples:
                keep_size = max_fim_samples // 2
                if self.fim_scores.numel() > keep_size:
                    keep_indices = torch.randperm(self.fim_scores.numel())[:keep_size]
                    self.fim_scores = self.fim_scores[keep_indices]
                
                remaining_size = max_fim_samples - self.fim_scores.numel()
                if new_samples.numel() > remaining_size:
                    new_indices = torch.randperm(new_samples.numel())[:remaining_size]
                    new_samples = new_samples[new_indices]
                
                self.fim_scores = torch.cat([self.fim_scores, new_samples])
            else:
                self.fim_scores = torch.cat([self.fim_scores, new_samples])
        
        threshold = torch.quantile(fim_flat.detach(), 1.0 - self.importance_threshold).detach()
        importance_mask = (fim_flat.detach() > threshold).detach()
        
        # Clean up temporary variables
        del fim_flat, fim_diag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return importance_mask.reshape(x.shape)

    def _compute_variance_importance(self, x):
        """Method 2: Local variance-based importance calculation"""
        x_detached = x.detach()
        
        if x_detached.dim() == 2:  # [batch, features]
            batch_size, features = x_detached.shape
            variance_map = torch.zeros_like(x_detached)
            
            window_size = min(5, features)  # Use larger window
            for i in range(features):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(features, i + window_size // 2 + 1)
                local_region = x_detached[:, start_idx:end_idx]
                if local_region.numel() > 1:
                    variance_map[:, i] = local_region.var(dim=1, keepdim=False)
                else:
                    variance_map[:, i] = 0.0
                    
        elif x_detached.dim() == 4:  # [batch, channels, height, width]
            # Use convolution to calculate local variance
            kernel_size = 3
            padding = kernel_size // 2
            
            mean_kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                                   device=x_detached.device) / (kernel_size * kernel_size)
            local_mean = F.conv2d(x_detached.mean(dim=1, keepdim=True), mean_kernel, padding=padding)
            
            x_squared = x_detached.pow(2).mean(dim=1, keepdim=True)
            local_mean_squared = F.conv2d(x_squared, mean_kernel, padding=padding)
            variance_map = local_mean_squared - local_mean.pow(2)
            variance_map = variance_map.expand_as(x_detached)
        else:
            variance_map = torch.full_like(x_detached, x_detached.var().item())

        variance_flat = variance_map.flatten().detach()
        
        if self.fast_mode:
            return self._compute_statistical_threshold_mask(variance_flat, x.shape)
        else:
            return self._compute_memory_efficient_variance_mask(variance_flat, x.shape)

    def _compute_magnitude_importance(self, x):
        """Method 3: Activation magnitude-based importance calculation"""
        magnitude_map = x.abs().detach()
        magnitude_flat = magnitude_map.flatten().detach()
        
        if self.fast_mode:
            return self._compute_statistical_threshold_mask(magnitude_flat, x.shape)
        else:
            return self._compute_memory_efficient_quantile_mask(magnitude_flat, x.shape)
    

    def _compute_statistical_threshold_mask(self, values_flat, original_shape):
        """Statistical threshold calculation (fast mode)"""
        mean_val = values_flat.mean()
        std_val = values_flat.std()
        
        # Adjust threshold multiplier based on importance_threshold
        threshold_multiplier = 2.5 * self.importance_threshold
        threshold = mean_val + threshold_multiplier * std_val
        
        importance_mask = (values_flat > threshold).detach()
        return importance_mask.reshape(original_shape)
    
    def _compute_memory_efficient_quantile_mask(self, values_flat, original_shape):
        """Memory-efficient quantile calculation"""
        max_samples = 50000
        
        if values_flat.numel() <= max_samples:
            # Small tensor direct calculation
            try:
                threshold = torch.quantile(values_flat, 1.0 - self.importance_threshold).detach()
                importance_mask = (values_flat > threshold).detach()
                return importance_mask.reshape(original_shape)
            except RuntimeError:
                return self._compute_statistical_threshold_mask(values_flat, original_shape)
        else:
            # Large tensor uses sampling method
            sample_indices = torch.randperm(values_flat.numel(), device=values_flat.device)[:max_samples]
            sampled_values = values_flat[sample_indices].detach()
            
            try:
                threshold = torch.quantile(sampled_values, 1.0 - self.importance_threshold).detach()
                importance_mask = (values_flat > threshold).detach()
                return importance_mask.reshape(original_shape)
            except RuntimeError:
                # Sampling method also fails, fall back to statistical method
                return self._compute_statistical_threshold_mask(values_flat, original_shape)
    
    def _compute_memory_efficient_variance_mask(self, variance_flat, original_shape):
        """Memory-efficient variance threshold calculation"""
        max_samples = 100000
        
        if variance_flat.numel() <= max_samples:
            # Small tensor direct statistical calculation
            try:
                variance_mean = variance_flat.mean()
                variance_std = variance_flat.std()
                threshold = variance_mean + 0.5 * variance_std
                importance_mask = (variance_flat > threshold).detach()
                return importance_mask.reshape(original_shape)
            except RuntimeError:
                # Statistical calculation fails, fall back to simpler method
                return self._compute_statistical_threshold_mask(variance_flat, original_shape)
        else:
            # Large tensor uses sampling method to calculate statistics
            sample_indices = torch.randperm(variance_flat.numel(), device=variance_flat.device)[:max_samples]
            sampled_variance = variance_flat[sample_indices].detach()
            
            try:
                variance_mean = sampled_variance.mean()
                variance_std = sampled_variance.std()
                threshold = variance_mean + 0.5 * variance_std
                importance_mask = (variance_flat > threshold).detach()
                return importance_mask.reshape(original_shape)
            except RuntimeError:
                # Sampling statistics also fails, fall back to simplest method
                return self._compute_statistical_threshold_mask(variance_flat, original_shape)

    def dalog_negative_reparameterization(self, x_neg):
        """DALog negative interval reparameterization"""
        x_neg_mapped = -x_neg
        return x_neg_mapped
    
    def forward(self, x):
        if self.n_bits == 32:
            return x
        
        if not self.inited:
            self.init_from(x)
        
        assert self.inited, "Quantizer must be initialized first"
        
        if not self.training and not self.training_mode:
            return self._inference_forward(x)
        
        # Activation preprocessing - can be controlled via config file
        if self.enable_preprocessing:
            x_clipped = self.gelu_adaptive_clipping(x)
        else:
            x_clipped = x
        
        # Importance calculation - unified use of compute_importance entry point
        importance_mask = self.compute_importance(x_clipped)
        
        # Hybrid quantization strategy
        x_high_importance = x_clipped * importance_mask
        x_low_importance = x_clipped * (~importance_mask)
        
        # High importance activations: DALog quantization
        if x_high_importance.sum() > 0:
            x_neg_mask = x_high_importance < 0
            x_pos_mask = x_high_importance >= 0
            
            x_neg_reparamed = self.dalog_negative_reparameterization(
                x_high_importance * x_neg_mask
            )
            
            x_pos_quantized = self.dalog_quantizer(x_high_importance * x_pos_mask)
            x_neg_quantized = -self.dalog_quantizer(x_neg_reparamed)
            
            x_high_quantized = x_pos_quantized + x_neg_quantized * x_neg_mask
        else:
            x_high_quantized = x_high_importance
            
        # Low importance activations: uniform quantization
        if x_low_importance.sum() > 0:
            x_low_quantized = self.uniform_quantizer(x_low_importance)
        else:
            x_low_quantized = x_low_importance
            
        result = x_high_quantized + x_low_quantized
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def _inference_forward(self, x):
        """Simplified forward propagation for inference stage"""
        # Activation preprocessing - can be controlled via config file
        if self.enable_preprocessing:
            x_clipped = self.gelu_adaptive_clipping(x)
        else:
            x_clipped = x
        
        # Inference stage unified use of compute_importance entry point
        importance_mask = self.compute_importance(x_clipped)
        
        # Subsequent quantization logic same as training stage
        x_high_importance = x_clipped * importance_mask
        x_low_importance = x_clipped * (~importance_mask)
        
        if x_high_importance.sum() > 0:
            x_neg_mask = x_high_importance < 0
            x_pos_mask = x_high_importance >= 0
            
            x_neg_reparamed = self.dalog_negative_reparameterization(
                x_high_importance * x_neg_mask
            )
            
            x_pos_quantized = self.dalog_quantizer(x_high_importance * x_pos_mask)
            x_neg_quantized = -self.dalog_quantizer(x_neg_reparamed)
            
            x_high_quantized = x_pos_quantized + x_neg_quantized * x_neg_mask
        else:
            x_high_quantized = x_high_importance
            
        if x_low_importance.sum() > 0:
            if not self.uniform_quantizer.inited:
                print(f"Warning: uniform_quantizer not initialized, performing emergency initialization...")
                scale = x_low_importance.abs().max()
                if self.sym:
                    uniform_scale = scale / (self.n_levels - 0.5)
                    self.uniform_quantizer.scale = nn.Parameter(uniform_scale.clone())
                else:
                    uniform_scale = scale / (2 * self.n_levels - 1)
                    zero_point = torch.zeros_like(scale)
                    self.uniform_quantizer.scale = nn.Parameter(uniform_scale.clone())
                    self.uniform_quantizer.zero_point = nn.Parameter(zero_point.clone())
                
                self.uniform_quantizer.drop_prob = self.drop_prob
                self.uniform_quantizer.inited = True
            
            x_low_quantized = self.uniform_quantizer(x_low_importance)
        else:
            x_low_quantized = x_low_importance
            
        return x_high_quantized + x_low_quantized

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, ' \
               f'channel_wise={self.channel_wise}, importance_method={self.importance_method.value})'
        

