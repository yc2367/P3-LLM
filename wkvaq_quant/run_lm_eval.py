# Import necessary modules
import argparse
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from lm_eval.models.utils import stop_sequences_criteria
from tqdm import tqdm
from loguru import logger
import os
import json

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

from utils import (
    load_model_and_tokenizer, 
    add_common_args, 
    add_quant_args, 
    get_quant_config,
    set_seed,
    model2path
)


def run_lm_eval(
    model, tokenizer, args, max_length=100
):
    model.seqlen = max_length
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=args.batch_size)
    task_manager = lm_eval.tasks.TaskManager()

    # Setting task_manager to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in lm_eval/tasks.
    # simple_evaluate will instantiate its own task_manager is the it is set to None here.
    logger.info(f"Evaluation Task(s): {args.tasks}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Limit: {args.limit}")

    #NOTE (Yuzong): use "confirm_run_unsafe_code" to support HumanEval dataset
    confirm_run_unsafe_code = False
    for task in args.tasks:
        if 'humaneval' in task.lower():
            import os
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            confirm_run_unsafe_code = True
            break

    with torch.no_grad():
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=args.tasks,
            task_manager=task_manager,
            limit=args.limit,
            log_samples=True,
            num_fewshot=args.num_fewshot,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            apply_chat_template=args.apply_chat_template,
            confirm_run_unsafe_code=confirm_run_unsafe_code
        ) 
    res = make_table(results)
    
    return results['results']


if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    # Set random seed
    set_seed(42)

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_quant_args(parser)
    parser.add_argument("--tasks", type=lambda s: [item for item in s.split(',')], default=[], help="Task to be evaled")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for lm_eval tasks")
    parser.add_argument("--limit", type=int, default=None, help="limit number of samples to run")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of shots for evaluation")
    parser.add_argument("--fewshot_as_multiturn", action="store_true", help="Whether to treat fewshot as multiturn or not.")
    parser.add_argument("--apply_chat_template", action="store_true", help="Whether to apply chat template or not.")
    parser.add_argument("--output_dir", type=str, default="results/lm_eval", help="output directory")
    args = parser.parse_args()  
    
    quant_config = get_quant_config(args)
    model_name = args.model_name
    model_name_or_path = model2path[model_name]

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    logger.info(f"#################### Model Info ####################")
    logger.info(f"* Model: {model_name_or_path}")
    logger.info(f"#################### Start evaluating LM_Eval with the following configurations: ####################")
    logger.info(f"* Bench compression!!!")
    logger.info(f"* Weights bits: {args.w_bits}")
    logger.info(f"* Weights group size: {args.w_group_size}")
    logger.info(f"* Activation bits: {args.a_bits}")
    logger.info(f"* Activation group size: {args.a_group_size}")
    logger.info(f"* Attn-Score bits: {args.p_bits}")
    logger.info(f"* Query bits: {args.q_bits}")
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
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.use_fp16:
        output_file_name = "Baseline_FP16"
    elif (args.a_bits == 16) and (args.k_bits == 16) and (args.v_bits == 16) and (args.q_bits == 16):
        output_file_name = f"Baseline-w{args.w_bits}-wgs_{args.w_group_size}-a16-k16-v16-p{args.p_bits}-q{args.q_bits}"
    elif (args.a_bits == 16) and (args.w_bits == 16) and (args.q_bits == 16):
        output_file_name = f"Baseline-w16-a16-q16-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    elif (args.a_bits == 16) and (args.q_bits == 16):
        output_file_name = f"Baseline-w{args.w_bits}-wgs_{args.w_group_size}-a16-q16-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    elif (args.q_bits == 16):
        output_file_name = f"Baseline-w{args.w_bits}-a{args.a_bits}-wgs_{args.w_group_size}-ags_{args.a_group_size}-q16-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    else:
        output_file_name = f"w{args.w_bits}-a{args.a_bits}-wgs_{args.w_group_size}-ags_{args.a_group_size}-q{args.q_bits}-{args.kv_quant_method}-k{args.k_bits}-v{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-p{args.p_bits}-scale_{args.apply_k_scale}"
    output_file_path = os.path.join(output_dir, f"{output_file_name}.json")
    # check if result file exists
    print(output_file_path)
    if os.path.isfile(output_file_path):
        print(f'Found existing output file {output_file_name} for this experiment. Exit!\n\n')
        exit()
    print(f'Results will be saved to the output file: {output_file_name}\n')

    logger.info("#################### Loading model and tokenizer ... ####################")
    model, tokenizer = load_model_and_tokenizer(model_name, quant_config=quant_config, use_fp16=args.use_fp16)
    
    logger.info("#################### Start running LM_Eval zero-shot evaluation ... #################### ")
    res = run_lm_eval(model, tokenizer, args)
    
    # Save results to JSON file
    with open(output_file_path, "w") as f:
        json.dump(res, f, indent=4)

    print(f"Results saved to {output_file_path} \n\n")
    