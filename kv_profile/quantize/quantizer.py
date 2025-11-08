import torch
from typing import Optional


@torch.no_grad()
def a_quant_per_group(
    x_fp: torch.Tensor, q_bits: int=8, group_size: int=-1
):
    """
    Symmetric per-group activation quantization.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    """
    if q_bits >= 16:
        return x_fp
    
    batch, seq_len, h_dim = x_fp.shape
    if group_size <= 0:
        num_groups = 1
        group_size = h_dim
    else:
        num_groups = h_dim // group_size
        assert num_groups * group_size == h_dim, \
            f"The input tensor's last dimension {x_fp.shape[-1]} is not divisible by group_size {group_size}"
    x_fp_new = x_fp.view(batch, seq_len, num_groups, group_size)

    # ############### INT8 Quantization ###############
    # qmax = 127
    # rmax = torch.amax(x_fp_new.abs(), dim=-1, keepdim=True)
    # rmax = rmax.clamp(min=1e-5)
    # scale = (rmax / qmax).clamp_(min=1e-6)

    # x_q  = torch.clamp(torch.round(x_fp_new / scale), min=-qmax, max=qmax)
    # x_dq = x_q * scale
    # x_dq = x_dq.view(batch, seq_len, h_dim)

    # return x_dq

    ############### FP8-E4M3 Quantization ###############
    qmax = 448
    rmax = torch.amax(x_fp_new.abs(), dim=-1, keepdim=True)
    scale = rmax / qmax
    scale = scale.clamp_(min=1e-6)
    x_scaled = (x_fp_new / scale).abs()

    x_dq_sign = torch.sign(x_fp_new)
    x_dq_exp  = (x_scaled + (x_scaled == 0).type(x_scaled.dtype)).log2().floor().clamp_(min=-6)
    x_dq_man  = torch.round(x_scaled / 2**x_dq_exp * 2**3) / 2**3

    x_dq = x_dq_sign * 2**x_dq_exp * x_dq_man * scale
    x_dq = x_dq.view(batch, seq_len, h_dim)

    return x_dq


@torch.no_grad()
def q_quant_per_head(
    x_fp: torch.Tensor, q_bits: int=8, group_size: int=-1
):
    """
    Symmetric per-group activation quantization.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    """
    if q_bits >= 16:
        return x_fp
    
    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp_new = x_fp.transpose(1, 2).view(batch, seq_len, -1)
    # x_fp_new = x_fp

    # ############### INT8 Quantization ###############
    # qmax = 127
    # rmax = torch.amax(x_fp_new.abs(), dim=-1, keepdim=True)
    # rmax = rmax.clamp(min=1e-5)
    # scale = (rmax / qmax).clamp_(min=1e-6)

    # x_q  = torch.clamp(torch.round(x_fp_new / scale), min=-qmax, max=qmax)
    # x_dq = x_q * scale

    # x_dq = x_dq.view(batch, seq_len, num_head, h_dim).transpose(1, 2)

    # return x_dq

    ############### FP8-E3M4 Quantization ###############
    # qmax = 30
    # rmax = torch.amax(x_fp_new.abs(), dim=-1, keepdim=True)
    # scale = rmax / qmax
    # scale = scale.clamp_(min=1e-6)
    # x_scaled = (x_fp_new / scale).abs()

    # x_dq_sign = torch.sign(x_fp_new)
    # x_dq_exp  = (x_scaled + (x_scaled == 0).type(x_scaled.dtype)).log2().floor().clamp_(min=-2)
    # x_dq_man  = torch.round(x_scaled / 2**x_dq_exp * 2**4) / 2**4

    # x_dq = x_dq_sign * 2**x_dq_exp * x_dq_man * scale
    # return x_dq

    ############### FP8-E4M3 Quantization ###############
    qmax = 448
    rmax = torch.amax(x_fp_new.abs(), dim=-1, keepdim=True)
    scale = rmax / qmax
    scale = scale.clamp_(min=1e-6)
    x_scaled = (x_fp_new / scale).abs()

    x_dq_sign = torch.sign(x_fp_new)
    x_dq_exp  = (x_scaled + (x_scaled == 0).type(x_scaled.dtype)).log2().floor().clamp_(min=-6)
    x_dq_man  = torch.round(x_scaled / 2**x_dq_exp * 2**3) / 2**3

    x_dq = x_dq_sign * 2**x_dq_exp * x_dq_man * scale

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).transpose(1, 2)

    return x_dq


@torch.no_grad()
def k_quant_per_token(
    x_fp: torch.Tensor, q_bits: int=4, group_size: int=128,
    apply_k_bias: Optional[bool]=False, k_bias: Optional[torch.Tensor]=None,
    apply_k_scale: Optional[bool]=False, k_scale: Optional[torch.Tensor]=None,
):
    """
    Asymmetric per-token key quantization.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    :param apply_k_bias: whether to apply per-channel bias subtraction
    :param k_bias: if apply_k_bias == True, then the input tensor's channel subtract k_bias
    :param apply_k_scale: whether to apply per-channel scaling
    :param k_scale: if apply_k_scale == True, then the input tensor's channel will be smoothed (divided) by k_scale
    """
    if q_bits == 16:
        return x_fp

    if apply_k_bias and (not apply_k_scale):
        x_fp_new = x_fp - k_bias
    elif apply_k_scale and (not apply_k_bias):
        x_fp_new = x_fp / k_scale
    elif apply_k_scale and apply_k_bias:
        x_fp_new = (x_fp - k_bias) / k_scale
    else:
        x_fp_new = x_fp
    
    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp_new = (
        x_fp_new.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )

    num_groups = (num_head * h_dim) // group_size
    assert num_groups * group_size == num_head * h_dim, \
        f"The input tensor's last dimension {x_fp_new.shape[-1]} is not divisible by group_size {group_size}"
    x_fp_new = x_fp_new.view(batch, seq_len, num_groups, group_size)

    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp_new, dim=-1, keepdim=True)
    rmax = torch.amax(x_fp_new, dim=-1, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = (torch.round(-rmin / scale_fp)).clamp_(qmin, qmax)
    x_q  = torch.clamp(torch.round(x_fp_new / scale_fp) + zeropoint, min=qmin, max=qmax)
    x_dq = (x_q - zeropoint) * scale_fp # de-quantized tensor

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3).contiguous()
    if apply_k_scale:
        x_dq = x_dq * k_scale
    if apply_k_bias:
        x_dq = x_dq + k_bias
    
    # print(f'Key Quant Error Pre-RoPE: {(x_dq - x_fp).to(torch.float32).pow(2).mean() * 1e3}')
    return x_dq


@torch.no_grad()
def k_quant_per_channel(
    x_fp: torch.Tensor, q_bits: int=4, group_size: int=128,
):
    """
    Asymmetric per-channel key quantization for KIVI.
    NOTE: The zero-point is in FP16.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    """
    if q_bits >= 16:
        return x_fp

    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp_new = (
        x_fp.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )
    #NOTE(Trial): See the influence of fp16
    #.float()
    num_groups = seq_len // group_size
    assert num_groups * group_size == seq_len, \
        f"The input tensor's sequence length {seq_len} is not divisible by group_size {group_size}"
    x_fp_new = x_fp_new.view(batch, num_groups, group_size, num_head * h_dim)
    
    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp_new, dim=-2, keepdim=True)
    rmax = torch.amax(x_fp_new, dim=-2, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = -rmin
    q_tensor = torch.clamp(torch.round((x_fp_new + zeropoint) / scale_fp), min=qmin, max=qmax)
    x_dq = (q_tensor * scale_fp) - zeropoint

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3).contiguous()
    #print(f'Quant Error: {(x_dq - x_fp).pow(2).mean()}')
    return x_dq


@torch.no_grad()
def v_quant_per_token(
    x_fp: torch.Tensor, q_bits: int=4, group_size: int=128
):
    """
    Asymmetric per-token value quantization.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    """
    batch, num_head, seq_len, h_dim = x_fp.shape
    if q_bits >= 16:
        return x_fp, torch.ones((batch, num_head, seq_len, 1), dtype=x_fp.dtype, device=x_fp.device)

    x_fp_new = (
        x_fp.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )
    #NOTE(Trial): See the influence of fp16
    #.float()

    num_groups = (num_head * h_dim) // group_size
    assert num_groups * group_size == num_head * h_dim, \
        f"The input tensor's last dimension {x_fp_new.shape[-1]} is not divisible by group_size {group_size}"
    x_fp_new = x_fp_new.view(batch, seq_len, num_groups, group_size)

    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp_new, dim=-1, keepdim=True)
    rmax = torch.amax(x_fp_new, dim=-1, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = (torch.round(-rmin / scale_fp)).clamp_(qmin, qmax)
    x_q = torch.clamp(torch.round(x_fp_new / scale_fp) + zeropoint, min=qmin, max=qmax)
    x_q = x_q - zeropoint

    scale_fp = scale_fp.view(batch, seq_len, num_head, h_dim // group_size).permute(0, 2, 1, 3).contiguous()
    x_q = x_q.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3).contiguous()

    return x_q, scale_fp


@torch.no_grad()
def p_quant_per_block(
    x_fp: torch.Tensor, q_bits: int=8
):
    """
    Asymmetric per-token value quantization.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    """
    assert q_bits in [8, 12, 16], \
        f'Invalid precision \"{q_bits}\" provided for attention score. Allowed precisions are {{8, 12, 16}}'
    
    x_fp = x_fp.to(torch.float16)

    if q_bits >= 16:
        return x_fp
    
    if q_bits == 12:
        x_fp_tmp = x_fp.view(torch.int16)
        x_fp_lsb = x_fp_tmp.bitwise_and(7)
        x_dq     = x_fp_tmp.bitwise_and(65528)

        roundup_mask = x_fp_lsb.gt(3)
        x_dq[roundup_mask] = x_dq[roundup_mask] + 8
        x_dq = x_dq.view(torch.float16)
        # print(f'Quant error P: {(x_dq - x_fp).pow(2).mean()}')
        return x_dq
    
    if q_bits == 8:
        x_fp_tmp = x_fp.view(torch.int16)
        x_fp_lsb = x_fp_tmp.bitwise_and(127)
        x_dq     = x_fp_tmp.bitwise_and(65408) 
            
        roundup_mask = x_fp_lsb.gt(63)
        x_dq[roundup_mask] = x_dq[roundup_mask] + 128
        x_dq = x_dq.view(torch.float16)
        # print(f'Quant error P: {(x_dq - x_fp).pow(2).sum()}')
        return x_dq

    rmax = torch.amax(x_fp.abs(), dim=-1, keepdim=True)
    qmax = 2**(q_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    x_q = torch.clamp(torch.round(x_fp / scale_fp), min=qmin, max=qmax)
    x_dq = x_q * scale_fp

    # print(f'Quant error P: {(x_dq - x_fp).pow(2).mean()}')

    return x_dq


def k_quant_function(
    k_fp, quant_config, 
    k_bias: Optional[torch.Tensor]=None,
    k_scale: Optional[torch.Tensor]=None,
):
    kv_quant_method = quant_config.kv_quant_method
    k_bits = quant_config.k_bits
    k_group_size = quant_config.k_group_size
    assert kv_quant_method in ['KTVT', 'KCVT'], \
        f'Invalid quantization method \"{kv_quant_method}\" provided. ' + \
        'Currently only support \"KTVT\" and \"KCVT\" quantization.'

    if kv_quant_method == "KTVT": # key per-token, value per-token
        k_dq = k_quant_per_token(
            k_fp, k_bits, k_group_size,
            apply_k_bias=quant_config.apply_k_bias, k_bias=k_bias,
            apply_k_scale=quant_config.apply_k_scale, k_scale=k_scale
        )
    elif kv_quant_method == "KCVT": # key per-channel, value per-token
        k_dq = k_quant_per_channel(
            k_fp, k_bits, k_group_size
        )       

    return k_dq


def v_quant_function(
    v_fp, quant_config, use_fp16: bool=False
):
    if use_fp16:
        v_bits = 16
    else:
        v_bits = quant_config.v_bits

    v_group_size = quant_config.v_group_size
    v_q, v_scale = v_quant_per_token(
        v_fp, v_bits, v_group_size
    )

    return v_q, v_scale


def quant_matmul_pv(
    p_fp, v_int, v_scale, quant_config, is_prefill: bool=False
):
    """
    p_fp:    batch, num_head, q_seq_len, kv_seq_len
    v_int:   batch, num_head, kv_seq_len, h_dim
    v_scale: batch, num_head, kv_seq_len, num_groups_per_head
    """
    if is_prefill:
        p_bits = quant_config.p_bits_pf
    else:
        p_bits = quant_config.p_bits_dc
    
    batch, num_head, kv_seq_len, h_dim = v_int.shape
    v_group_size = quant_config.v_group_size
    if (v_group_size is None) or (v_group_size <= 0):
        v_group_size = h_dim
    num_groups_per_head = h_dim // v_group_size
    q_seq_len = p_fp.shape[-2]
    
    v_int = v_int.view(batch, num_head, kv_seq_len, num_groups_per_head, v_group_size)
    v_int = v_int.transpose(2, 3).contiguous() # batch, num_head, num_groups_per_head, kv_seq_len, group_size
    p_fp = p_fp[:, :, None, :, :].expand(-1, -1, num_groups_per_head, -1, -1) # batch, num_head, num_groups_per_head, q_seq_len, kv_seq_len
    v_scale = v_scale.transpose(2, 3).unsqueeze(-2) # batch, num_head, num_groups_per_head, 1, kv_seq_len
    p_fp = p_fp * v_scale

    p_dq = p_quant_per_block(p_fp, p_bits) # batch, num_head, num_groups_per_head, q_seq_len, kv_seq_len
    attn_output = torch.matmul(p_dq, v_int) # batch, num_head, num_groups_per_head, q_seq_len, group_size
    attn_output = attn_output.transpose(2, 3).contiguous().view(batch, num_head, q_seq_len, h_dim) # batch, num_head, q_seq_len, h_dim
    
    return attn_output
