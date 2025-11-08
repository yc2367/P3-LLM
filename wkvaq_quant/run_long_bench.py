import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import random
import argparse
os.environ["WANDB_DISABLED"] = "true"
from loguru import logger

from transformers import LlamaConfig, MistralConfig, AutoTokenizer

from utils import (
    load_model_and_tokenizer, 
    add_common_args, 
    add_quant_args, 
    get_quant_config,
    set_seed,
    model2path
)
from longbench_utils import scorer, model2maxlen, dataset2prompt, dataset2maxlen


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    # Update: we add the template for the new llama-3-instruct model
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds


if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    # Set random seed
    set_seed(42)

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_quant_args(parser)
    parser.add_argument("--output_dir", type=str, default="results/long_bench", help="output directory")
    args = parser.parse_args()

    quant_config = get_quant_config(args)
    model_name = args.model_name
    model_name_or_path = model2path[model_name]
    max_length = model2maxlen[model_name]

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    logger.info(f"#################### Model Info ####################")
    logger.info(f"* Model: {model_name_or_path}")
    logger.info(f"#################### Start evaluating ppl with the following configurations: ####################")
    logger.info(f"* Bench compression!!!")
    logger.info(f"* Weights bits: {args.w_bits}")
    logger.info(f"* Weights group size: {args.w_group_size}")
    logger.info(f"* Activation bits: {args.a_bits}")
    logger.info(f"* Activation group size: {args.a_group_size}")
    logger.info(f"* Attn-Score bits: {args.p_bits}")
    logger.info(f"* KV-cache quantization method: {args.kv_quant_method}")
    logger.info(f"* Post-Attn KV Quant?: {args.kv_quant_post_attn}")
    logger.info(f"* Post-RoPE Key Quant?: {args.k_quant_post_rope}")
    logger.info(f"* Key bits: {args.k_bits}")
    logger.info(f"* Value bits: {args.v_bits}")
    logger.info(f"* Key group size: {args.k_group_size}")
    logger.info(f"* Value group size: {args.v_group_size}")
    logger.info(f"* KV residual length: {args.kv_residual_len}")
    logger.info(f"* Apply key bias?: {args.apply_k_bias}")
    logger.info(f"* Apply key scale?: {args.apply_k_scale}")

    logger.info("#################### Creating output directory ... ####################")
    if args.use_fp16:
        output_config_dir = "Baseline_FP16"
    elif (args.a_bits == 16) and (args.k_bits == 16) and (args.v_bits == 16) and (args.q_bits == 16):
        output_config_dir = f"Baseline-w{args.w_bits}-wgs_{args.w_group_size}-a16-k16-v16-p{args.p_bits}-q{args.q_bits}"
    elif (args.a_bits == 16) and (args.w_bits == 16) and (args.q_bits == 16):
        output_config_dir = f"Baseline-w16-a16-q16-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    elif (args.a_bits == 16) and (args.q_bits == 16):
        output_config_dir = f"Baseline-w{args.w_bits}-wgs_{args.w_group_size}-a16-q16-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    elif (args.q_bits == 16):
        output_config_dir = f"Baseline-w{args.w_bits}-a{args.a_bits}-wgs_{args.w_group_size}-ags_{args.a_group_size}-q16-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    else:
        output_config_dir = f"w{args.w_bits}-a{args.a_bits}-wgs_{args.w_group_size}-ags_{args.a_group_size}-q{args.q_bits}-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    output_dir = os.path.join(args.output_dir, model_name, output_config_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("#################### Loading model and tokenizer ... ####################")
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, quant_config=quant_config, max_memory=args.max_memory, use_slow_attn=True, use_fp16=args.use_fp16)

    logger.info("#################### Start running LongBench evaluation ... ####################")
    scores = dict()
    datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', dataset, split='test')

        # check if result file exists
        output_pred_file_path = os.path.join(output_dir, f"{dataset}_pred.jsonl")
        print(output_pred_file_path)
        if os.path.isfile(output_pred_file_path):
            print(f'Found existing output file {output_config_dir}/{dataset}_pred.jsonl for this experiment. Continue to the next dataset!')
            continue
        
        # running evaluation
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name)
        with open(output_pred_file_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False, indent=4)
                f.write('\n')

        # calculate score
        predictions, answers, lengths = [], [], []
        for pred in preds:
            predictions.append(pred["pred"])
            answers.append(pred["answers"])
            if "length" in pred:
                lengths.append(pred["length"])
            all_classes = pred["all_classes"]
        score = scorer(dataset, predictions, answers, all_classes)
        logger.info(f"#################### Dataset: {dataset} ####################")
        logger.info(f"#################### Score: {score} ####################")
        scores[dataset] = score
        output_res_file_path = os.path.join(output_dir, f"{dataset}_result.json")
        with open(output_res_file_path, "w") as f:
            data_to_log = {"dataset": dataset, "score": score}
            json.dump(data_to_log, f, indent=4)
    
    # store full results
    output_res_file_path = os.path.join(output_dir, f"longbench_full_result.json")
    with open(output_res_file_path, "w") as f:
        json.dump(scores, f, indent=4)