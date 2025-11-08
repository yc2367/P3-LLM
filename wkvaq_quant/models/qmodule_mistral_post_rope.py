import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.mistral.configuration_mistral import *
from transformers.models.mistral.modeling_mistral import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from quantize.quant_config import QuantConfig
from quantize.quantizer import a_quant_function, q_quant_function, k_quant_function, v_quant_function, quant_matmul_pv

from typing import Optional, Tuple
import logging, warnings
logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


class QuantMistralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig, quant_config: QuantConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # KV-cache quantization config
        self.quant_config = quant_config
        self.kv_quant_post_attn = quant_config.kv_quant_post_attn
        self.kv_residual_len = quant_config.kv_residual_len
        self.apply_k_bias = quant_config.apply_k_bias
        self.apply_k_scale = quant_config.apply_k_scale

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        #NOTE (Yuzong): activation quantization
        hidden_states_quant = a_quant_function(hidden_states, self.quant_config)

        #########################  QKV Projection  #########################
        query_states = self.q_proj(hidden_states_quant)
        key_states = self.k_proj(hidden_states_quant)
        value_states = self.v_proj(hidden_states_quant)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Update sequence length
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        
        cos, sin = self.rotary_emb(key_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        #NOTE (Yuzong): quantized self-attention
        if not self.kv_quant_post_attn:
            if past_key_value is None: # Prefill Stage

                # Calculate scale and bias for key-smoothing
                if self.apply_k_bias and (not self.apply_k_scale):
                    key_quant_bias = key_states.mean(dim=2, keepdim=True)
                    key_quant_scale = None
                elif self.apply_k_scale and (not self.apply_k_bias):
                    key_quant_scale = key_states.abs().amax(dim=2, keepdim=True)
                    key_quant_bias = None
                elif self.apply_k_scale and self.apply_k_bias:
                    key_quant_bias = key_states.mean(dim=2, keepdim=True)
                    key_quant_scale = (key_states - key_quant_bias).abs().amax(dim=2, keepdim=True).pow(0.6)
                else:
                    key_quant_scale = None
                    key_quant_bias = None
                
                ######################### Divide KV-cache into Quantized and Float parts #########################
                if key_states.shape[-2] <= self.kv_residual_len:
                    key_states_quant = None
                    key_states_float = key_states
                else:
                    key_states_quant = key_states[:, :, :-self.kv_residual_len, :]
                    key_states_float = key_states[:, :, -self.kv_residual_len:, :]
                
                if value_states.shape[-2] <= self.kv_residual_len:
                    value_states_quant = None
                    value_states_float = value_states
                else:
                    value_states_quant = value_states[:, :, :-self.kv_residual_len, :]
                    value_states_float = value_states[:, :, -self.kv_residual_len:, :]

                ####################################### Quantize KV-cache ######################################
                if (key_states_quant is not None) and (value_states_quant is not None):
                    key_states_quant = k_quant_function(
                        key_states_quant, self.quant_config, 
                        k_bias=key_quant_bias, k_scale=key_quant_scale
                    )  
                    value_states_quant_int, value_states_quant_scale = v_quant_function(
                        value_states_quant, self.quant_config
                    ) 
                else:
                    key_states_quant = None
                    value_states_quant_int = None
                    value_states_quant_scale = None

                ############################################ Q x K.T ############################################
                # Fuse query quantization with per-channel key smoothing
                if self.apply_k_scale:
                    query_states_scaled = query_states * repeat_kv(key_quant_scale, self.num_key_value_groups)
                else:
                    query_states_scaled = query_states
                query_states_quant = q_quant_function(query_states_scaled, self.quant_config)

                # if self.quant_config.q_bits < 16:
                #     print(f'Q quant error: {(query_states_quant - query_states_scaled).pow(2).mean()}')

                if key_states_quant is None:
                    attn_weights_quant = None
                    attn_weights_float = torch.matmul(query_states, repeat_kv(key_states_float, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn_weights = attn_weights_float
                else:
                    attn_weights_quant = torch.matmul(query_states_quant, repeat_kv(key_states_quant, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn_weights_float = torch.matmul(query_states, repeat_kv(key_states_float, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn_weights = torch.cat([attn_weights_quant, attn_weights_float], dim=-1)

                if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                    )

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)

                ############################################# P x V ############################################
                if value_states_quant_int is None:
                    value_states_full = repeat_kv(value_states_float, self.num_key_value_groups)
                    attn_output = torch.matmul(attn_weights, value_states_full) 
                else:  # value_states_float will never be None in this case
                    value_states_float_len = value_states_float.shape[-2]
                    value_states_full_int = repeat_kv(value_states_quant_int, self.num_key_value_groups)
                    value_states_full_scale = repeat_kv(value_states_quant_scale, self.num_key_value_groups)
                    value_states_full_float = repeat_kv(value_states_float, self.num_key_value_groups)
                    attn_weights_quant = attn_weights[..., :-value_states_float_len]
                    attn_weights_float = attn_weights[..., -value_states_float_len:]
                    attn_output_quant = quant_matmul_pv(
                        attn_weights_quant,
                        value_states_full_int, value_states_full_scale,
                        self.quant_config,  # is_prefill=True
                    ) 
                    attn_output_float = torch.matmul(attn_weights_float, value_states_full_float) 
                    attn_output = attn_output_quant + attn_output_float
            else: # Decoding Stage

                ############################## Prepare Quantized and Float KV-cache ##############################
                key_states_quant = past_key_value[0] # quantized key
                key_states_float = past_key_value[1] # full-precision residual key
                value_states_quant_int = past_key_value[2] # quantized value integer
                value_states_quant_scale = past_key_value[3] # quantized value scale
                value_states_float = past_key_value[4] # full-precision residual value
                key_quant_bias = past_key_value[5] # per-channel bias for key quantization
                key_quant_scale = past_key_value[6] # per-channel scale for key quantization

                key_states_float = torch.cat([key_states_float, key_states], dim=2)
                value_states_float = torch.cat([value_states_float, value_states], dim=2)
                key_states_float_len = key_states_float.shape[-2]
                value_states_float_len = value_states_float.shape[-2]
                assert key_states_float_len == value_states_float_len, \
                    f"key_states_float_len and value_states_float_len should be equal !"
                
                ####################################### Quantize KV-cache ######################################
                if value_states_float_len > self.kv_residual_len:
                    assert value_states_float_len == self.kv_residual_len + 1, \
                        f"Wrong value_states_float_len !"

                    value_states_quant_int_new, value_states_quant_scale_new = v_quant_function(
                        value_states_float[:, :, :1, :], self.quant_config
                    ) 
                    if value_states_quant_int is None:
                        value_states_quant_int = value_states_quant_int_new
                        value_states_quant_scale = value_states_quant_scale_new
                    else:
                        value_states_quant_int = torch.cat([value_states_quant_int, value_states_quant_int_new], dim=2)
                        value_states_quant_scale = torch.cat([value_states_quant_scale, value_states_quant_scale_new], dim=2)
                    value_states_float = value_states_float[:, :, 1:, :]

                    key_states_quant_new = k_quant_function(
                        key_states_float[:, :, :1, :], self.quant_config, 
                        k_bias=key_quant_bias, k_scale=key_quant_scale
                    )
                    if key_states_quant is None:
                        key_states_quant = key_states_quant_new
                    else:
                        key_states_quant = torch.cat([key_states_quant, key_states_quant_new], dim=2)
                    key_states_float = key_states_float[:, :, 1:, :]

                    assert value_states_float.shape[-2] == self.kv_residual_len, \
                        f"Wrong value_states_float_len !"
                    assert key_states_float.shape[-2] == self.kv_residual_len, \
                        f"Wrong key_states_float_len !"
                
                ######################################## Q x K.T ########################################
                # Fuse query quantization with per-channel key smoothing
                if self.apply_k_scale and (not self.apply_k_bias):
                    query_states_scaled = query_states * key_quant_scale
                else:
                    query_states_scaled = query_states
                query_states_quant = q_quant_function(query_states_scaled, self.quant_config)

                if key_states_quant is None:
                    attn_weights_quant = None
                    attn_weights_float = torch.matmul(query_states, repeat_kv(key_states_float, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn_weights = attn_weights_float
                else:
                    attn_weights_quant = torch.matmul(query_states_quant, repeat_kv(key_states_quant, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn_weights_float = torch.matmul(query_states, repeat_kv(key_states_float, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn_weights = torch.cat([attn_weights_quant, attn_weights_float], dim=-1)

                if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                    )

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                
                ######################################## P x V #######################################
                if value_states_quant_int is None:
                    value_states_full = repeat_kv(value_states_float, self.num_key_value_groups)
                    attn_output = torch.matmul(attn_weights, value_states_full) 
                else:  # value_states_float will never be None in this case
                    value_states_float_len = value_states_float.shape[-2]
                    value_states_full_int = repeat_kv(value_states_quant_int, self.num_key_value_groups)
                    value_states_full_scale = repeat_kv(value_states_quant_scale, self.num_key_value_groups)
                    value_states_full_float = repeat_kv(value_states_float, self.num_key_value_groups)
                    attn_weights_quant = attn_weights[..., :-value_states_float_len]
                    attn_weights_float = attn_weights[..., -value_states_float_len:]
                    attn_output_quant = quant_matmul_pv(
                        attn_weights_quant,
                        value_states_full_int, value_states_full_scale,
                        self.quant_config,  # is_prefill=False
                    ) 
                    attn_output_float = torch.matmul(attn_weights_float, value_states_full_float) 
                    attn_output = attn_output_quant + attn_output_float
                    
        else: # Quantize KV-cache after the self-attention operation
            if past_key_value is None: # Prefill Stage

                # Calculate scale and bias for key-smoothing
                if self.apply_k_bias and (not self.apply_k_scale):
                    key_quant_bias = key_states.mean(dim=2, keepdim=True)
                    key_quant_scale = None
                elif self.apply_k_scale and (not self.apply_k_bias):
                    key_quant_scale = key_states.abs().amax(dim=2, keepdim=True)
                    key_quant_bias = None
                elif self.apply_k_scale and self.apply_k_bias:
                    key_quant_bias = key_states.mean(dim=2, keepdim=True)
                    key_quant_scale = (key_states - key_quant_bias).abs().amax(dim=2, keepdim=True).pow(0.6)
                else:
                    key_quant_scale = None
                    key_quant_bias = None
                
                ######################### Divide KV-cache into Quantized and Float parts #########################
                if key_states.shape[-2] <= self.kv_residual_len:
                    key_states_quant = None
                    key_states_float = key_states
                else:
                    key_states_quant = key_states[:, :, :-self.kv_residual_len, :]
                    key_states_float = key_states[:, :, -self.kv_residual_len:, :]
                
                if value_states.shape[-2] <= self.kv_residual_len:
                    value_states_quant = None
                    value_states_float = value_states
                else:
                    value_states_quant = value_states[:, :, :-self.kv_residual_len, :]
                    value_states_float = value_states[:, :, -self.kv_residual_len:, :]
                
                ############################################ Q x K.T ############################################
                key_states_full = repeat_kv(key_states_full, self.num_key_value_groups)
                #NOTE The "v_quant_function" here is not doing any quantization by setting "use_fp16=True".
                # It simply returns a format that allows multiplying with quantized attn-score.
                value_states_int, value_states_scale = v_quant_function(
                    value_states, self.quant_config, use_fp16=True
                ) 
                value_states_int = repeat_kv(value_states_int, self.num_key_value_groups)
                value_states_scale = repeat_kv(value_states_scale, self.num_key_value_groups)

                if self.config.use_slow_attn: #NOTE To avoid OOM during Attention in LongBench evaluation
                    #NOTE the number of heads processed in every iteration is hard-coded "4"
                    assert self.num_heads % 4 == 0, \
                        f"The number of attention heads = {self.num_heads} in this model is not divisible by 4 !"
                    
                    attn_output = torch.zeros_like(query_states) 

                    ############################################ Q x K.T ############################################
                    for i_h in range(0, self.num_heads // 4): #NOTE the number of heads processed in every iteration is hard-coded "4"
                        attn_weights = torch.matmul(query_states[:, i_h*4 : (i_h+1)*4, :, :], key_states_full[:, i_h*4 : (i_h+1)*4, :, :].transpose(2, 3)) / math.sqrt(self.head_dim)
                        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                                f" {attn_weights.size()}"
                            )
                        if attention_mask is not None:
                            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                                raise ValueError(
                                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                                )
                            attn_weights = attn_weights + attention_mask
                            attn_weights = torch.max(
                                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                            )

                        # upcast attention to fp32
                        attn_weights = nn.functional.softmax(
                            attn_weights, dim=-1, dtype=torch.float32
                        ).to(query_states.dtype)

                        ############################################# P x V ############################################
                        attn_output[:, i_h*4 : (i_h+1)*4, :, :] = quant_matmul_pv(
                            attn_weights, 
                            value_states_int[:, i_h*4 : (i_h+1)*4, :, :], value_states_scale[:, i_h*4 : (i_h+1)*4, :, :],
                            self.quant_config,  # is_prefill=True
                        ) 
                else:
                    ############################################ Q x K.T ############################################
                    attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) / math.sqrt(self.head_dim)
                    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                            f" {attn_weights.size()}"
                        )
                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights + attention_mask
                        attn_weights = torch.max(
                            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                        )

                    # upcast attention to fp32
                    attn_weights = nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)

                    ############################################# P x V ############################################
                    attn_output = quant_matmul_pv(
                        attn_weights, 
                        value_states_int, value_states_scale,
                        self.quant_config,  # is_prefill=True
                    ) 

                ####################################### Quantize KV-cache ######################################
                if (key_states_quant is not None) and (value_states_quant is not None):
                    key_states_quant = k_quant_function(
                        key_states_quant, self.quant_config, 
                        k_bias=key_quant_bias, k_scale=key_quant_scale
                    )  
                    value_states_quant_int, value_states_quant_scale = v_quant_function(
                        value_states_quant, self.quant_config
                    ) 
                else:
                    key_states_quant = None
                    value_states_quant_int = None
                    value_states_quant_scale = None
            else: # Decoding Stage

                ############################## Prepare Quantized and Float KV-cache ##############################
                key_states_quant = past_key_value[0] # quantized key
                key_states_float = past_key_value[1] # full-precision residual key
                value_states_quant_int = past_key_value[2] # quantized value integer
                value_states_quant_scale = past_key_value[3] # quantized value scale
                value_states_float = past_key_value[4] # full-precision residual value
                key_quant_bias = past_key_value[5] # per-channel bias for key quantization
                key_quant_scale = past_key_value[6] # per-channel scale for key quantization

                key_states_float = torch.cat([key_states_float, key_states], dim=2)
                value_states_float = torch.cat([value_states_float, value_states], dim=2)
                key_states_float_len = key_states_float.shape[-2]
                value_states_float_len = value_states_float.shape[-2]
                assert key_states_float_len == value_states_float_len, \
                    f"key_states_float_len and value_states_float_len should be equal !"
                
                ####################################### Quantize KV-cache ######################################
                if value_states_float_len > self.kv_residual_len:
                    assert value_states_float_len == self.kv_residual_len + 1, \
                        f"Wrong value_states_float_len !"

                    value_states_quant_int_new, value_states_quant_scale_new = v_quant_function(
                        value_states_float[:, :, :1, :], self.quant_config
                    ) 
                    if value_states_quant_int is None:
                        value_states_quant_int = value_states_quant_int_new
                        value_states_quant_scale = value_states_quant_scale_new
                    else:
                        value_states_quant_int = torch.cat([value_states_quant_int, value_states_quant_int_new], dim=2)
                        value_states_quant_scale = torch.cat([value_states_quant_scale, value_states_quant_scale_new], dim=2)
                    value_states_float = value_states_float[:, :, 1:, :]

                    key_states_quant_new = k_quant_function(
                        key_states_float[:, :, :1, :], self.quant_config, 
                        k_bias=key_quant_bias, k_scale=key_quant_scale
                    )
                    if key_states_quant is None:
                        key_states_quant = key_states_quant_new
                    else:
                        key_states_quant = torch.cat([key_states_quant, key_states_quant_new], dim=2)
                    key_states_float = key_states_float[:, :, 1:, :]

                    assert value_states_float.shape[-2] == self.kv_residual_len, \
                        f"Wrong value_states_float_len !"
                    assert key_states_float.shape[-2] == self.kv_residual_len, \
                        f"Wrong key_states_float_len !"
                
                ######################################## Q x K.T ########################################
                if key_states_quant is None:
                    key_states_full = key_states_float
                else:
                    key_states_full = torch.cat([key_states_quant, key_states_float], dim=2)

                key_states_full = repeat_kv(key_states_full, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                    )
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                
                ######################################## P x V #######################################
                if value_states_quant_int is None:
                    value_states_full = repeat_kv(value_states_float, self.num_key_value_groups)
                    attn_output = torch.matmul(attn_weights, value_states_full) 
                else:  # value_states_float will never be None in this case
                    value_states_float_len = value_states_float.shape[-2]
                    value_states_full_int = repeat_kv(value_states_quant_int, self.num_key_value_groups)
                    value_states_full_scale = repeat_kv(value_states_quant_scale, self.num_key_value_groups)
                    value_states_full_float = repeat_kv(value_states_float, self.num_key_value_groups)
                    attn_weights_quant = attn_weights[..., :-value_states_float_len]
                    attn_weights_float = attn_weights[..., -value_states_float_len:]
                    attn_output_quant = quant_matmul_pv(
                        attn_weights_quant,
                        value_states_full_int, value_states_full_scale,
                        self.quant_config,  # is_prefill=False
                    ) 
                    attn_output_float = torch.matmul(attn_weights_float, value_states_full_float) 
                    attn_output = attn_output_quant + attn_output_float

        past_key_value = (
            key_states_quant, key_states_float,
            value_states_quant_int, value_states_quant_scale, value_states_float, 
            key_quant_bias, key_quant_scale,
            kv_seq_len
        ) if use_cache else None

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        #NOTE (Yuzong): activation quantization
        attn_output_quant = a_quant_function(attn_output, self.quant_config)
        attn_output = self.o_proj(attn_output_quant)

        attn_weights = None
        return attn_output, attn_weights, past_key_value


class QuantMistralMLP(nn.Module):
    def __init__(self, config, quant_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        #NOTE (Yuzong): quantization config
        self.quant_config = quant_config

    def forward(self, hidden_states):
        #NOTE (Yuzong): activation quantization
        hidden_states_quant = a_quant_function(hidden_states, self.quant_config)

        intermediate_states = self.act_fn(self.gate_proj(hidden_states_quant)) * self.up_proj(hidden_states_quant)
        #NOTE (Yuzong): activation quantization
        intermediate_states_quant = a_quant_function(intermediate_states, self.quant_config)
        down_proj = self.down_proj(intermediate_states_quant)

        return down_proj


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, quant_config: QuantConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantMistralAttention(config=config, quant_config=quant_config)
        self.mlp = QuantMistralMLP(config=config, quant_config=quant_config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
            

class QuantMistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig, quant_config: QuantConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MistralDecoderLayer(config, quant_config) for _ in range(config.num_hidden_layers)])

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, 
                (batch_size, seq_length), 
                inputs_embeds, 
                past_key_values_length,
                sliding_window=self.config.sliding_window
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QuantMistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MistralConfig, quant_config: QuantConfig):
        super().__init__(config)
        self.model = QuantMistralModel(config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = past_key_values.to_legacy_cache()
            if len(past_key_values) == 0:
                past_key_values = None
        if past_key_values is not None:
            past_length = past_key_values[0][-1]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
