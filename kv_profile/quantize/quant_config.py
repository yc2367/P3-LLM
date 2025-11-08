class QuantConfig(dict):
    def __init__(
        self,
        # general quantization parameters
        w_bits: int=16,
        a_bits: int=16,
        q_bits: int=16,
        k_bits: int=16,
        v_bits: int=16,
        p_bits_pf: int=16, # attn-score precision dduring prefill 
        p_bits_dc: int=16, # attn-score precision dduring decode 
        w_group_size: int=-1,
        a_group_size: int=-1,
        q_group_size: int=-1,
        k_group_size: int=-1,
        v_group_size: int=-1,
        # KV-cache quantization config
        kv_quant_method: str="KTVT",
        kv_residual_len: int=1,
        kv_quant_post_attn: bool=False, # If True, KV-cache will be quantized after self-attention
        apply_k_bias: bool=False,
        apply_k_scale: bool=False,
    ):
        for nbits in [w_bits, k_bits, v_bits]:
            assert (nbits is None) or (nbits in [2, 3, 4, 6, 8, 16]), \
                f'Invalid precision \"{nbits}\" provided for weight / KV-cache. Allowed precisions are {{4, 6, 8, 16}}'
        for nbits in [a_bits, q_bits]:
            assert (nbits is None) or (nbits in [8, 16]), \
                f'Invalid precision \"{nbits}\" provided for activation / query. Allowed precisions are {{8, 16}}'
        for nbits in [p_bits_pf, p_bits_dc]:
            assert (nbits is None) or (nbits in [8, 12, 16]), \
                f'Invalid precision \"{nbits}\" provided for attention-score. Allowed precisions are {{8, 12, 16}}'
        
        for group_size in [w_group_size, a_group_size, q_group_size, k_group_size, v_group_size]:
            assert (group_size is None) or (group_size <= 0) or (group_size in [32, 64, 128]), \
                f'Invalid precision \"{nbits}\" provided for activation / query. Allowed precisions are {{8, 16}}'
        
        if (kv_quant_method == 'KCVT') and (kv_residual_len > 0) and (k_group_size > 0):
            assert kv_residual_len % k_group_size == 0, \
                f"The KV residual length {kv_residual_len} should be a multiple of the key group size {k_group_size}"
            assert (not apply_k_bias) and (not apply_k_scale), \
                 f'For KCVT quantization. \"apply_k_bias\" and \"apply_k_scale\" must be set to False'

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.q_bits = q_bits
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.p_bits_pf = p_bits_pf
        self.p_bits_dc = p_bits_dc
        self.w_group_size = w_group_size
        self.a_group_size = a_group_size
        self.q_group_size = q_group_size
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size
        self.p_group_size = -1 # don't apply group-wise quantization for attention-score

        # KV-cache quantization config
        self.kv_quant_method = kv_quant_method
        self.kv_quant_post_attn = kv_quant_post_attn
        self.kv_residual_len = kv_residual_len
        self.apply_k_bias = apply_k_bias
        self.apply_k_scale = apply_k_scale
    
    def __repr__(self):
        return repr(self.__dict__)
