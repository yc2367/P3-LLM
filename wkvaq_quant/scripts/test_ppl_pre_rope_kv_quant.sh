#!/bin/bash

########## Modify the path according to your HOME directory ##########
HOME_DIR="/home/yc2367/llm/P2-LLM/wkvaq_quant"
AWQ_DIR="/share/abdelfattah/temp_yc2367/awq_quant_model"
######################################################################

seq_len=2048
OUTPUT_DIR=${HOME_DIR}/results/ppl_pre_rope_quant_${seq_len}

model_list=("llama-7b" "llama-13b" "llama-2-7b" "llama-2-13b")

dataset_list="wikitext,c4"

k_bits_list=(4)
v_bits_list=(4)
k_group_size_list=(128)
v_group_size_list=(128)

p_bits_list=(8)

w_bits_list=(4)
w_group_size_list=(128)
wq_dtype="bitmod"

a_bits=(8)
a_group_size_list=(-1)


for model_name in "${model_list[@]}"
do
    for k_bits in "${k_bits_list[@]}"
    do
        for v_bits in "${v_bits_list[@]}"
        do
            for k_group_size in "${k_group_size_list[@]}"
            do
                for v_group_size in "${v_group_size_list[@]}"  
                do
                    for w_bits in "${w_bits_list[@]}"
                    do
                        for w_group_size in "${w_group_size_list[@]}"
                        do
                            for p_bits in "${p_bits_list[@]}"
                            do
                                for a_group_size in "${a_group_size_list[@]}"  
                                do
                                    if [[ ${wq_dtype} == "bitmod" ]]
                                    then
                                        awq_model_path_lp=${AWQ_DIR}/${model_name}/w${w_bits}-g${w_group_size}-bitmod
                                    else 
                                        awq_model_path_lp=${AWQ_DIR}/${model_name}/w${w_bits}-g${w_group_size}
                                    fi
                                    
                                    rm /home/yc2367/llm/P2-LLM/data_cache/testloader_mistral_7b_c4*

                                    ####################  All FP16  ####################
                                    # python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                    #     --datasets ${dataset_list} --seq_len ${seq_len} \
                                    #     --output_dir ${OUTPUT_DIR} --use_fp16
                                    
                                    ####################  KTVT  ####################
                                    python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                        --datasets ${dataset_list} --seq_len ${seq_len} --output_dir ${OUTPUT_DIR} \
                                        --kv_quant_method "KTVT" \
                                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                                        --p_bits ${p_bits} \
                                        --w_bits ${w_bits} --w_group_size ${w_group_size} \
                                        --awq_model_path_lp ${awq_model_path_lp} \
                                        --a_bits ${a_bits} --a_group_size ${a_group_size}
                                    
                                    python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                        --datasets ${dataset_list} --seq_len ${seq_len} --output_dir ${OUTPUT_DIR} \
                                        --kv_quant_method "KTVT" --apply_k_scale \
                                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                                        --p_bits ${p_bits} \
                                        --w_bits ${w_bits} --w_group_size ${w_group_size} \
                                        --awq_model_path_lp ${awq_model_path_lp} \
                                        --a_bits ${a_bits} --a_group_size ${a_group_size}
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done