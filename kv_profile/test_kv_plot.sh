#!/bin/bash

model="llama-3.1-8b"
dataset_list="wikitext"


HOME_DIR="/home/yc2367/llm/P2-LLM/kv_profile"

k_bits=16
v_bits=16


####################  KTVT  ####################
python ${HOME_DIR}/run_ppl.py --model_name ${model} \
    --datasets ${dataset_list} --seq_len 4096 \
    --k_bits ${k_bits} --v_bits ${v_bits} \
    --kv_quant_method "KTVT" \
    --output_dir ${HOME_DIR}/results/ppl/ 
