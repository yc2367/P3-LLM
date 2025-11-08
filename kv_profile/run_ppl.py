import torch
import torch.nn as nn

from datasets import load_dataset

import argparse
from tqdm import tqdm
from loguru import logger
import os
import json
import random
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

    
@torch.no_grad()
def eval_ppl(model, tokenizer, args, device="cuda"):
    results = {}
    for task_eval in args.datasets:
        if task_eval == "wikitext":
            # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seq_len = args.seq_len
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seq_len
            nlls = []
            loss_fct = nn.CrossEntropyLoss()
            for i in tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seq_len) : ((i + 1) * model.seq_len)
                ][:, 1:]
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seq_len
                nlls.append(neg_log_likelihood.item())

            ppl = torch.exp(torch.tensor(nlls).sum() / (nsamples * model.seq_len))
            print(f'Wikitext-2 perplexity: {ppl.item()}')
            print('\n')

            results["wikitext"] = ppl.item()
        elif task_eval == "c4":
            model_net = model_name_or_path.split('/')[-1]
            model_family = '_'.join(model_net.lower().split('-')[:-1])
            model.seq_len = args.seq_len

            cache_testloader = f'/home/yc2367/llm/P2-LLM/data_cache/testloader_{model_family}_c4_{args.seq_len}.cache'
            os.makedirs(os.path.dirname(cache_testloader), exist_ok=True)
            if os.path.exists(cache_testloader):
                testenc = torch.load(cache_testloader)
                print(f"load calibration from {cache_testloader}")
            else:
                valenc = []
                testenc = load_dataset("allenai/c4", data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split="validation")
                for _ in range(256): # run 256 samples
                    while True:
                        i = random.randint(0, len(testenc) - 1)
                        tmp = tokenizer(testenc[i]['text'], return_tensors='pt')
                        if tmp.input_ids.shape[1] > (model.seq_len+1):
                            break
                    i = random.randint(0, tmp.input_ids.shape[1] - model.seq_len - 1)
                    j = i + model.seq_len
                    valenc.append(tmp.input_ids[:, i:j])
                testenc = torch.hstack(valenc)
                torch.save(testenc, cache_testloader)
            
            nsamples = testenc.numel() // model.seq_len
            loss_fct = nn.CrossEntropyLoss()
            nlls = []
            with tqdm(range(nsamples)) as progress:
                for i in progress:
                    batch = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)].to(model.device)
                    with torch.no_grad():
                        lm_logits = model(batch, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)][:, 1:].to(model.device)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * model.seq_len
                    nlls.append(neg_log_likelihood.item())
                    progress.set_description(f"Evaluating")

            ppl = torch.exp(torch.tensor(nlls).sum() / (nsamples * model.seq_len))
            print(f'C4 perplexity: {ppl.item()}')
            print('\n')

            results['c4'] = ppl.item()
    return results
    

if __name__ == '__main__':
    set_seed(0)
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_quant_args(parser)
    parser.add_argument('--datasets', type=lambda s: [item for item in s.split(',')], default=['wikitext'], help="Task to be evaled")
    parser.add_argument('--seq_len', type=int, help='sequence length for ppl evaluation', default=2048)
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    parser.add_argument("--output_dir", type=str, default="results/ppl", help="output directory")
    args = parser.parse_args()  
    
    quant_config = get_quant_config(args)
    model_name = args.model_name
    model_name_or_path = model2path[model_name]

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    logger.info(f"#################### Model Info ####################")
    logger.info(f"* Model: {model_name_or_path}")
    logger.info(f"* Datasets: {args.datasets}")
    logger.info(f"* sequence length {args.seq_len}")
    logger.info(f"#################### Start evaluating ppl with the following configurations: ####################")
    logger.info(f"* Bench compression!!!")
    logger.info(f"* Attn-Score bits during Prefill: {args.p_bits_pf}")
    logger.info(f"* KV-cache quantization method: {args.kv_quant_method}")
    logger.info(f"* Key bits: {args.k_bits}")
    logger.info(f"* Value bits: {args.v_bits}")
    logger.info(f"* Key group size: {args.k_group_size}")
    logger.info(f"* Value group size: {args.v_group_size}")
    logger.info(f"* Quantize KV post Attn?: {args.kv_quant_post_attn}")
    logger.info(f"* KV residual length: {args.kv_residual_len}")
    logger.info(f"* Apply key bias?: {args.apply_k_bias}")
    logger.info(f"* Apply key scale?: {args.apply_k_scale}")

    logger.info("#################### Creating output directory ... ####################")
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"{args.kv_quant_method}-kbits_{args.k_bits}-vbits_{args.v_bits}-kgs_{args.k_group_size}-vgs_{args.v_group_size}-res_{args.kv_residual_len}-pbitspf_{args.p_bits_pf}-pbitsdc_{args.p_bits_dc}-bias_{args.apply_k_bias}-scale_{args.apply_k_scale}"
    output_file_path = os.path.join(output_dir, f"{output_file_name}.txt")
    # check if result file exists
    print(output_file_path)
    if os.path.isfile(output_file_path):
        print(f'Found existing output file {output_file_name} for this experiment. Exit!')
        exit()
    
    logger.info("#################### Loading model and tokenizer ... ####################")
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, quant_config=quant_config, use_fp16=args.use_fp16)

    logger.info("#################### Start running perplexity evaluation ... ####################")
    res = eval_ppl(model, tokenizer, args)

    # Save results to JSON file
    with open(output_file_path, "w") as f:
        for dataset, ppl in res.items():
            logger.info(f"{dataset} PPL: {ppl}")
            f.write(f"{dataset.ljust(10)} PPL: {ppl}\n")
    
    print(f"Results saved to {output_file_path}")