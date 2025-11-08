#!/bin/bash

model=("llama-3.2-3b")
dataset_list="wikitext,c4"


HOME_DIR="/home/yc2367/llm/P2-LLM/kv_quant"

k_bits_list=(4 3)
v_bits_list=(4 3)
k_group_size_list=(64 128)
v_group_size_list=(64 128)

k_bits_list=(4)
v_bits_list=(3)
k_group_size_list=(64)
v_group_size_list=(32)
p_bits_list=(16 12 8)

for k_bits in "${k_bits_list[@]}"
do
    for v_bits in "${v_bits_list[@]}"
    do
        for k_group_size in "${k_group_size_list[@]}"
        do
            for v_group_size in "${v_group_size_list[@]}"  
            do
                for p_bits in "${p_bits_list[@]}"  
                do
                    ####################  FP16  ####################
                    python ${HOME_DIR}/run_ppl.py --model_name ${model} \
                        --datasets ${dataset_list} --seq_len 2048 \
                        --use_fp16 \
                        --output_dir ${HOME_DIR}/results/ppl/ \

                    ####################  KTVT  ####################
                    python ${HOME_DIR}/run_ppl.py --model_name ${model} \
                        --datasets ${dataset_list} --seq_len 2048 \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                        --kv_quant_method "KTVT" \
                        --output_dir ${HOME_DIR}/results/ppl/ \

                    python ${HOME_DIR}/run_ppl.py --model_name ${model} \
                        --datasets ${dataset_list} --seq_len 2048 \
                        --kv_quant_method "KTVT" \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                        --output_dir ${HOME_DIR}/results/ppl/ \
                        --apply_k_bias \

                    python ${HOME_DIR}/run_ppl.py --model_name ${model} \
                        --datasets ${dataset_list} --seq_len 2048 \
                        --kv_quant_method "KTVT" \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                        --output_dir ${HOME_DIR}/results/ppl/ \
                        --apply_k_scale \
                    
                    ####################  KCVT  ####################
                    kv_residual_len=64
                    if [ ${k_group_size} -eq 64 ]; then
                        kv_residual_len=128;
                    fi
                    python ${HOME_DIR}/run_ppl.py --model_name ${model} \
                        --datasets ${dataset_list} --seq_len 2048 \
                        --kv_quant_method "KCVT" --kv_residual_len ${kv_residual_len} \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size $(( k_group_size*2 )) --v_group_size ${v_group_size} \
                        --output_dir ${HOME_DIR}/results/ppl/ 
                done
            done
        done
    done
done