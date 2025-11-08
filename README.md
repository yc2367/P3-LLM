# P3-LLM: Efficient Mixed-Precisioin & Mixed-Format W4A8KV4 LLM Quantization

## 1. Getting Started
Clone the repository and its 3rd-party submodules, including AWQ and LM-Evaluation-Harness.
```
git clone --recurse-submodules https://github.com/yc2367/P3-LLM.git
```
The quantization code base is inside the `wkvaq_quant` directory. 

## 2. Obtaining AWQ quantized LLM
First, set up AWQ by followinig the instructions [here](https://github.com/yc2367/P3-LLM/tree/tinychat_1.5).

Refer to `wkvaq_quant/scripts/awq/run_awq.sh` and perform AWQ weight quantization. At the top, change `HOME_DIR` to your AWQ directory. The string variable `wq_dtype` can be "int" or "bitmod", where the latter is a state-of-the-art 4-bit data type, as described in the [BitMoD paper](https://arxiv.org/abs/2411.11745). The list variable `w_bit_list` and `group_size_list` contains the weight precision and group_size that you want to use. They are 4-bit and 128 by default.

After performing AWQ, you can run `run_awq_save_4b_model.sh` to evaluate the perplexity of quantized 4-bit model and save it. At the top, change `AWQ_DIR` to the directory where you want to save the fake-quantized model.

## 3. Evalaute mixed-precision LLM
Go to `wkvaq_quant/scripts/test_ppl_template.sh` and run Wikitext-2 and C4 perplexity evaluation. Currently, we only support Llama and Mistral models.

Change different quantization parameters by refering to their definition in `wkvaq_quant/utils.py`. Specifically, 
- `--kv_quant_method`: "KTVT" by default, which adopts per-token head KV-cache quantization. It can also take "KCVT", which uses per-channel key quantization as described [here](https://arxiv.org/abs/2402.02750).
- `--kv_residual_len`: Number of most recent tokens that are maintained in FP16 during KV-cache quantization. By default, this is 1, i.e., all KV-cache is quantized. Setting to a higher value will result in better accuracy.
- `--apply_k_scale`: If set, then use our proposed dynamic per-channel key-cache smoothing. 
- `--k_quant_post_rope`: If set, then quantize key cache after RoPE, else quantize key cache before RoPE.
- `--p_bits`: the precision of attention score.
