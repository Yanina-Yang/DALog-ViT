import numpy as np
import math
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from itertools import product
from quantizers.logarithm import *
from quantizers.uniform import *
from quantizers.hybrid_adalog import *
from datetime import datetime


class MinMaxQuantMatMul(nn.Module):
    """Matrix Multiplication base class"""
    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.mode = mode
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = False)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
    
    def forward(self, A, B):
        if self.mode == 'raw':
            out = A @ B
        elif self.mode == "quant_forward":
            out = self.quant_forward(A, B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input_A(self, x):
        return self.A_quantizer(x)
    
    def quant_input_B(self, x):
        return self.B_quantizer(x)
    
    def quant_forward(self, A, B):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        return self.quant_input_A(A) @ self.quant_input_B(B)
    
    
class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    - Q @ K:
        - A's shape: B,H,S,C
        - B's shape: B,H,C,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,C
    """
    def __init__(self, A_bit=8, B_bit=8, mode="raw", metric="mse", search_round=1, eq_n=100, 
                 head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, mode)
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = head_channel_wise)
        self.metric = metric
        self.search_round = search_round
        self.eq_n = eq_n
        # the head dim is always dim-1
        self.head_channel_wise = head_channel_wise
        self.token_channel_wise = token_channel_wise
        self.num_heads = num_heads
        
        if not self.head_channel_wise:
            target_shape = [1, 1, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        else:
            target_shape = [1, self.num_heads, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
    
    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        if metric == "mae":
            similarity = -torch.abs(tensor_raw - tensor_sim)
        elif metric == "mse":
            similarity = -(tensor_raw - tensor_sim) ** 2
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity
        
    
class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw", metric="mse", calib_batch_size=32, 
                 search_round=1, eq_n=100, head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, mode, metric, search_round, eq_n, head_channel_wise, token_channel_wise, num_heads)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = self.raw_input[0].shape[0]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (4 * self.raw_input[0][:self.calib_size].numel()+
                 4 * self.raw_input[1][:self.calib_size].numel()+
                 8 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        # Prevent division by zero error: ensure parallel_eq_n is at least 1
        if self.parallel_eq_n <= 0:
            self.parallel_eq_n = 1
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        
        
class AsymmetricallyBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw", metric="mse", calib_batch_size=32, 
                 search_round=1, eq_n=100, head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, mode, metric, calib_batch_size, search_round, 
                         eq_n, head_channel_wise, token_channel_wise, num_heads)
        del self.A_quantizer, self.B_quantizer
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = False, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = False, channel_wise = head_channel_wise)
        if not self.head_channel_wise:
            target_shape = [1, 1, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.A_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
        else:
            target_shape = [1, self.num_heads, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.A_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
    
    def _search_best_A_scale(self, A_scale_candidates, A_zero_point_candidates):
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,b,*,dim2,dim3
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize A
                cur_A_scale = A_scale_candidates[p_st:p_ed]
                cur_A_zero_point = A_zero_point_candidates[p_st:p_ed]
                A_sim = A.squeeze(0)
                A_quant = ((A_sim / cur_A_scale).round_() + cur_A_zero_point).clamp(0, 2 * self.A_quantizer.n_levels - 1)
                A_sim = (A_quant - cur_A_zero_point).mul_(cur_A_scale) # shape: (parallel_eq_n,b,*,dim1,dim2)
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = torch.mean(similarity, dim=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1, 1, -1, 1, 1)
        tmp_A_scale = torch.gather(A_scale_candidates, dim=0, index=best_index)
        tmp_A_zero_point = torch.gather(A_zero_point_candidates, dim=0, index=best_index)
        self.A_quantizer.scale.data.copy_(tmp_A_scale.view(self.A_quantizer.scale.shape))
        self.A_quantizer.zero_point.copy_(tmp_A_zero_point.view(self.A_quantizer.zero_point.shape))
        return best_index
        
    def _search_best_B_scale(self, B_scale_candidates, B_zero_point_candidates):
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            A_sim = self.quant_input_A(A).unsqueeze(0) # shape: 1,b,*,dim1,dim2
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize B
                cur_B_scale = B_scale_candidates[p_st:p_ed]
                cur_B_zero_point = B_zero_point_candidates[p_st:p_ed]
                B_sim = B.squeeze(0)
                B_quant = ((B_sim / cur_B_scale).round_() + cur_B_zero_point).clamp(0, 2 * self.B_quantizer.n_levels - 1)
                B_sim = (B_quant - cur_B_zero_point).mul_(cur_B_scale) # shape: (parallel_eq_n,b,*,dim2,dim3)
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = torch.mean(similarity, dim=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1, 1, -1, 1, 1)
        tmp_B_scale = torch.gather(B_scale_candidates, dim=0, index=best_index)
        tmp_B_zero_point = torch.gather(B_zero_point_candidates, dim=0, index=best_index)
        self.B_quantizer.scale.data.copy_(tmp_B_scale.view(self.B_quantizer.scale.shape))
        self.B_quantizer.zero_point.copy_(tmp_B_zero_point.view(self.B_quantizer.zero_point.shape))
        return best_index
    
    def calculate_percentile_candidates(self, x, l=0.99, r=0.99999):
        percentiles_uppers, percentiles_lowers = [], []
        pct = torch.tensor([l, r])
        tensor_too_large = True
        mini_batch_size = 1
        if self.head_channel_wise:
            x_ = x.transpose(0, 1).contiguous() # shape: heads,b,*,dim1,dim2
            x_ = x_.view(x_.shape[0], mini_batch_size, -1) 
        else:
            x_ = x.view(1, mini_batch_size, -1)
        while tensor_too_large:
            try:
                uppers_candidates = torch.quantile(x_, pct.to(x_.device), dim=-1).mean(dim=-1, keepdim=False) # shape: 2,(heads or 1)
                lowers_candidates = torch.quantile(x_, (1 - pct).to(x_.device), dim=-1).mean(dim=-1, keepdim=False) # shape: 2,(heads or 1)
                tensor_too_large = False
            except:
                mini_batch_size *= 2
                x_ = x_.view(x_.shape[0], mini_batch_size, -1) if self.head_channel_wise else x_.view(1, mini_batch_size, -1)
        u_splits = torch.linspace(0, 1, steps=self.eq_n+1).cuda()[:, None, None, None, None] * (uppers_candidates[1] - uppers_candidates[0]).view(1, 1, -1, 1, 1)
        d_splits = torch.linspace(0, 1, steps=self.eq_n+1).cuda()[:, None, None, None, None] * (lowers_candidates[0] - lowers_candidates[1]).view(1, 1, -1, 1, 1)
        upper_candidates = uppers_candidates[0].view(1, 1, -1, 1, 1) + u_splits
        lower_candidates = lowers_candidates[1].view(1, 1, -1, 1, 1) + d_splits
        return upper_candidates, lower_candidates
        
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        A_uppers_candidates, A_lowers_candidates = self.calculate_percentile_candidates(self.raw_input[0].cuda(), l=0.99, r=0.99999)
        B_uppers_candidates, B_lowers_candidates = self.calculate_percentile_candidates(self.raw_input[1].cuda(), l=0.99, r=0.99999)
        A_scale_candidates = ((A_uppers_candidates - A_lowers_candidates) / (2 * self.A_quantizer.n_levels - 1)).contiguous().cuda()
        A_zero_point_candidates = -(A_lowers_candidates / A_scale_candidates).round().contiguous().cuda()
        B_scale_candidates = ((B_uppers_candidates - B_lowers_candidates) / (2 * self.B_quantizer.n_levels - 1)).contiguous().cuda()
        B_zero_point_candidates = -(B_lowers_candidates / B_scale_candidates).round().contiguous().cuda()
        self.A_quantizer.scale.data.copy_(A_scale_candidates[-2])
        self.A_quantizer.zero_point.data.copy_(A_zero_point_candidates[-2])
        self.B_quantizer.scale.data.copy_(B_scale_candidates[-2])
        self.B_quantizer.zero_point.data.copy_(B_zero_point_candidates[-2])
        self.A_quantizer.inited = True
        self.B_quantizer.inited = True
        
        A_best_index = self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
        B_best_index = self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        for e in range(self.search_round):
            if self.A_quantizer.n_bits < 32:
                for ee in range(2):
                    if ee % 2 == 0:
                        A_uppers_candidates_ = torch.gather(A_uppers_candidates, dim=0, index=A_best_index)
                        A_lowers_candidates_ = A_lowers_candidates
                    else:
                        A_uppers_candidates_ = A_uppers_candidates
                        A_lowers_candidates_ = torch.gather(A_lowers_candidates, dim=0, index=A_best_index)
                    A_scale_candidates = ((A_uppers_candidates_ - A_lowers_candidates_) / (2 * self.A_quantizer.n_levels - 1)).contiguous().cuda()
                    A_zero_point_candidates = -(A_lowers_candidates_ / A_scale_candidates).round().contiguous().cuda()
                    torch.cuda.empty_cache()
                    A_best_index = self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
            if self.B_quantizer.n_bits < 32:
                for ee in range(2):
                    if ee % 2 == 0:
                        B_uppers_candidates_ = torch.gather(B_uppers_candidates, dim=0, index=B_best_index)
                        B_lowers_candidates_ = B_lowers_candidates
                    else:
                        B_uppers_candidates_ = B_uppers_candidates
                        B_lowers_candidates_ = torch.gather(B_lowers_candidates, dim=0, index=B_best_index)
                    B_scale_candidates = ((B_uppers_candidates_ - B_lowers_candidates_) / (2 * self.B_quantizer.n_levels - 1)).contiguous().cuda()
                    B_zero_point_candidates = -(B_lowers_candidates_ / B_scale_candidates).round().contiguous().cuda()
                    torch.cuda.empty_cache()
                    B_best_index = self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        
        if self.token_channel_wise:
            BA, HA, NA, MA = self.raw_input[0].shape
            BB, HB, NB, MB = self.raw_input[1].shape
            assert BA == BB and HA == HB and MA == NB
            A_token_wise_scale = self.A_quantizer.scale.expand(-1, -1, NA, -1)
            B_token_wise_scale = self.B_quantizer.scale.expand(-1, -1, -1, MB)
            del self.A_quantizer.scale, self.B_quantizer.scale
            self.A_quantizer.scale = nn.Parameter(A_token_wise_scale.clone())
            self.B_quantizer.scale = nn.Parameter(B_token_wise_scale.clone())
        
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None
        

class PostSoftmaxAsymmetricallyBatchingQuantMatMul(AsymmetricallyBatchingQuantMatMul):
    """
    Quantization class for matrix multiplication after Softmax activation function
    """
    
    def __init__(self, A_bit=8, B_bit=8, mode="raw", metric="mse", calib_batch_size=32,
                 search_round=1, eq_n=100, head_channel_wise=True, token_channel_wise=False, 
                 num_heads=12, importance_threshold=0.5,):
        super().__init__(A_bit, B_bit, mode, metric, calib_batch_size, search_round,
                         eq_n, head_channel_wise, token_channel_wise, num_heads)

        del self.A_quantizer
       
        self.A_quantizer = DALogQuantizer(
            n_bits=A_bit, 
            symmetric=False,  # Softmax output is non-negative, use asymmetric quantization
            channel_wise=head_channel_wise,
        )

        # 重新初始化A_quantizer的scale参数形状
        if not self.head_channel_wise:
            target_shape = [1, 1, 1, 1]
        else:
            target_shape = [1, self.num_heads, 1, 1]
        self.A_quantizer.register_buffer('scale', torch.zeros(*target_shape))

    def quant_input_A(self, x):
        assert torch.all(x >= 0), "Softmax输出应为非负张量"
        return self.A_quantizer(x)

    def quant_input_B(self, x):
        return self.B_quantizer(x)

    def _search_best_A_scale_adalog(self, A_scale_candidates):
        batch_similarities = []
        
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            
            B_sim = self.quant_input_B(B).unsqueeze(0)  # shape: 1,b,*,dim2,dim3
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_A_scale = A_scale_candidates[p_st:p_ed]
                original_scale = self.A_quantizer.scale.clone()
                
                A_sim_list = []
                for i in range(cur_A_scale.shape[0]):
                    # 严格匹配形状
                    scale_to_copy = cur_A_scale[i].view(self.A_quantizer.scale.shape)
                    self.A_quantizer.scale.data.copy_(scale_to_copy)
                    A_sim = self.A_quantizer(A)
                    A_sim_list.append(A_sim.unsqueeze(0))
                
                self.A_quantizer.scale.data.copy_(original_scale)
                A_sim = torch.cat(A_sim_list, dim=0)  # shape: (parallel_eq_n,b,*,dim1,dim2)
                out_sim = A_sim @ B_sim  # shape: parallel_eq_n,b,*,dim1,dim3
                
                # 计算相似度并适配维度
                similarity = self._get_similarity(raw_out, out_sim, self.metric)
                dims = list(range(3, len(similarity.shape))) if self.head_channel_wise else list(range(2, len(similarity.shape)))
                similarity = torch.mean(similarity, dim=dims)
                similarity = similarity.sum(dim=1, keepdim=True)
                
                similarities.append(similarity)
            
            similarities = torch.cat(similarities, 0)  # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False)
        
        # Determine the best index and adapt shape
        if self.head_channel_wise:
            best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1, 1, -1, 1, 1)
        else:
            best_index = torch.argmax(batch_similarities, dim=0, keepdim=True).view(1, 1, 1, 1, 1)
        
        # 更新A_quantizer.scale
        tmp_A_scale = torch.gather(A_scale_candidates, dim=0, index=best_index)
        self.A_quantizer.scale.data.copy_(tmp_A_scale.view(self.A_quantizer.scale.shape))
        return best_index

    def calculate_softmax_scale_candidates(self, x, l=1e-6, r=1.0):
        """
        Reference calculate_percentile_candidates style, only generate scale_candidates
        For Softmax output [0,1] characteristics, generate candidate scales based on quantiles
        """
        num_scale = self.eq_n  # Total number of candidates (controlled by eq_n consistent with reference method)
        percentiles = []
        tensor_too_large = True
        mini_batch_size = 1  # Initial batch size to avoid memory explosion

        # Adapt to multi-head scenario: adjust tensor shape (heads dimension first)
        if self.head_channel_wise:
            x_ = x.transpose(0, 1).contiguous()  # shape: [heads, batch, ...]
            x_ = x_.view(x_.shape[0], mini_batch_size, -1)  # Flatten to [heads, mini_batch, features]
        else:
            x_ = x.view(1, mini_batch_size, -1)  # Global mode: [1, mini_batch, features]

        # Calculate quantiles in batches (avoid direct calculation of large tensors)
        while tensor_too_large:
            try:
                # Softmax output is non-negative, only need to calculate upper bound quantiles (l to r interval, e.g. 0.9-1.0)
                pct = torch.linspace(l, r, num_scale, device=x_.device)  # Generate num_scale quantile points
                percentiles = torch.quantile(x_, pct, dim=-1).mean(dim=-1, keepdim=False)  # Calculate quantiles by feature dimension, then average batches
                tensor_too_large = False  # Calculation successful, exit loop
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 若内存不足，增大 mini_batch_size 减少单批次特征数
                    mini_batch_size *= 2
                    if self.head_channel_wise:
                        x_ = x_.view(x_.shape[0], mini_batch_size, -1)
                    else:
                        x_ = x_.view(1, mini_batch_size, -1)
                else:
                    raise e

        # 生成 scale 候选值并适配形状
        if self.head_channel_wise:
            # 多头部场景：[num_scale, 1, heads, 1, 1]
            scale_candidates = percentiles.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            # 全局场景：[num_scale, 1, 1, 1, 1]
            scale_candidates = percentiles.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # Ensure scale is not 0 (Softmax output is non-negative, lower bound set to minimum value)
        scale_candidates = torch.clamp(scale_candidates, min=l, max=r)
        return scale_candidates

    def hyperparameter_searching(self):
        """
        Rewrite hyperparameter search, optimized for Softmax characteristics
        """
        self._initialize_calib_parameters()
        
        # Generate specialized scale candidates for A matrix (Softmax output)
        A_scale_candidates = self.calculate_softmax_scale_candidates(self.raw_input[0].cuda())
        
        # Use standard method for B matrix (if quantization is needed)
        B_uppers_candidates, B_lowers_candidates = self.calculate_percentile_candidates(
            self.raw_input[1].cuda(), l=0.99, r=0.99999)
        B_scale_candidates = ((B_uppers_candidates - B_lowers_candidates) / 
                             (2 * self.B_quantizer.n_levels - 1)).contiguous().cuda()
        B_zero_point_candidates = -(B_lowers_candidates / B_scale_candidates).round().contiguous().cuda()
        
        # 初始化B_quantizer
        self.B_quantizer.scale.data.copy_(B_scale_candidates[-2].view(self.B_quantizer.scale.shape))
        self.B_quantizer.zero_point.data.copy_(B_zero_point_candidates[-2].view(self.B_quantizer.zero_point.shape))
        self.B_quantizer.inited = True
        
        # 初始化A_quantizer - 严格匹配形状
        init_scale = A_scale_candidates[-2].view(self.A_quantizer.scale.shape)
        self.A_quantizer.scale.data.copy_(init_scale)
        self.A_quantizer.inited = True
        
        # 搜索最佳A_quantizer参数
        for e in range(self.search_round):
            A_best_index = self._search_best_A_scale_adalog(A_scale_candidates)
        
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'A_bit={self.A_quantizer.n_bits}, '
                f'B_bit={self.B_quantizer.n_bits}, '
                f'mode={self.mode}, '
                f'metric={self.metric}, '
                f'head_channel_wise={self.head_channel_wise}, '
                )

