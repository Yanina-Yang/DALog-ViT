class Config:
    def __init__(self):
        self.optim_size = 1024
        self.calib_size = 32
        self.optim_batch_size = 32
        self.calib_batch_size = 32
        
        self.w_bit = 3  # Weight quantization bits
        self.a_bit = 3  # Activation quantization bits
        self.ps_bit = 3
        self.qconv_a_bit = 8  # Convolution layer activation quantization bits
        self.qhead_a_bit = 3  # Head activation quantization bits
        
        # Calibration metric settings
        self.calib_metric = 'mse'
        self.matmul_head_channel_wise = True
        self.token_channel_wise = True
        self.eq_n = 128
        self.search_round = 3
        
        # Optimization settings
        self.keep_gpu = True
        self.optim_metric = 'fisher_dplr'  # Use enhanced Fisher information matrix
        self.temp = 20
        
        # Fisher settings
        self.k = 15
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        
        # QDrop settings
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5

        self.use_gelu_adaptive = True  
        self.importance_threshold = 0.5   # High/low importance threshold
        self.importance_method = "variance" 
        self.fast_mode = False 
        
        # Reconstruction optimization parameters
        self.reconstruction_weight = 0.01
        self.reconstruction_iters = 20000
        self.reconstruction_lr = 4e-5
        self.b_range = (20, 2)
        self.warmup = 0.2

config = Config()
