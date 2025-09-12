#!/bin/bash


HOME_DIR="/home/yc2367/llm/P2-LLM/3rdparty/llm-awq"
AWQ_DIR="/share/abdelfattah/temp_yc2367/awq_quant_model"

model_name_list=("llama-7b" "llama-13b" "llama-2-7b" "llama-2-13b" "llama-3.1-8b" "llama-3.2-3b" "llama-3.1-8b-ins" "llama-3.2-3b-ins" "mistral-7b-v1" "mistral-7b-v3") 
model_name_list=( "llama-3.2-3b" )

wq_dtype="int"
w_bit_list=(4)
group_size_list=(128 )


for model_name in "${model_name_list[@]}"
do
    if [[ ${model_name} == "llama-7b" ]]
    then
        model_path="huggyllama/llama-7b"
    elif [[ ${model_name} == "llama-13b" ]]
    then
        model_path="huggyllama/llama-13b"
    elif [[ ${model_name} == "llama-2-7b" ]]
    then
        model_path="meta-llama/Llama-2-7b-hf"
    elif [[ ${model_name} == "llama-2-13b" ]]
    then
        model_path="meta-llama/Llama-2-13b-hf"
    elif [[ ${model_name} == "llama-3.1-8b" ]]
    then
        model_path="meta-llama/Llama-3.1-8B"
    elif [[ ${model_name} == "llama-3.2-3b" ]]
    then
        model_path="meta-llama/Llama-3.2-3B"
    elif [[ ${model_name} == "llama-3.1-8b-ins" ]]
    then
        model_path="meta-llama/Llama-3.1-8B-Instruct"
    elif [[ ${model_name} == "llama-3.2-3b-ins" ]]
    then
        model_path="meta-llama/Llama-3.2-3B-Instruct"
    elif [[ ${model_name} == "mistral-7b-v1" ]]
    then
        model_path="mistralai/Mistral-7B-v0.1"
    elif [[ ${model_name} == "mistral-7b-v3" ]]
    then
        model_path="mistralai/Mistral-7B-v0.3"
    fi

    for w_bit in "${w_bit_list[@]}"
    do
        for group_size in "${group_size_list[@]}"
        do
            if [[ ${wq_dtype} == "bitmod" ]]
            then
                awq_cache_path=${HOME_DIR}/awq_cache/${model_name}-w${w_bit}-g${group_size}-bitmod.pt
                fake_quant_save_path=${AWQ_DIR}/${model_name}/w${w_bit}-g${group_size}-bitmod
            else 
                awq_cache_path=${HOME_DIR}/awq_cache/${model_name}-w${w_bit}-g${group_size}.pt
                fake_quant_save_path=${AWQ_DIR}/${model_name}/w${w_bit}-g${group_size}
            fi

            echo 
            echo 
            echo "#################### Running Experiment ####################"
            echo "Model name            = ${model_name}"
            echo "Model path            = ${model_path}"
            echo "Quant precision       = ${w_bit}"
            echo "Quant group size      = ${group_size}"
            echo "AWQ cache path        = ${awq_cache_path}"
            echo "Fake quant save path  = ${fake_quant_save_path}"
            echo "############################################################"
            echo 

            cd ${HOME_DIR}
            python -m awq.entry --model_path ${model_path} \
                --w_bit ${w_bit} --q_group_size ${group_size} --wq_dtype ${wq_dtype} \
                --load_awq ${awq_cache_path} --use_double_quant \
                --dump_fake ${fake_quant_save_path} \
                --tasks "wikitext"
        done
    done
done
