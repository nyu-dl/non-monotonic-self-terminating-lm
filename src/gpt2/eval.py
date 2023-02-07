import argparse
import os
import torch
import torch.nn.functional as F
import pprint
import wandb
import json
import pickle
from transformers import GPT2Tokenizer
from tqdm import tqdm
from tqdm.utils import _term_move_up
from collections import defaultdict
import data as data
import utils as utils
from train import validate

def parse_args_and_eval(input_args):
    args = parse_args(input_args)
    print(f"\n{'='*20} Evalutaion args: {'='*20}")
    print(pprint.pformat(vars(args)))
    
    # -- initialize weights and biases for expr tracking
    wandb.init(
        project=f"strlm-gpt2-eval-{args.decoding_method}",
        entity="eugenechoi",
        config=args,job_type="evalutaion",
        tags="baseline",
        name=args.expr_name,
        notes="prof.cho dr. lee project.",
        dir=args.expr_name,
        mode="online" if args.wandb == 1 else "disabled"
    )
    
    # -- set seed for reproducible expr
    utils.set_seed(args.seed)

    # -- setup device 
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    # -- load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    
    # -- special token ids
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    
    # -- load pretrained model
    model = utils.load_model(args)
    model.to(device)
    
    # -- load dataloaders 
    eval_dataloader, decode_dataloader = data.load_dataloaders(args, model.config, pad_token_id, eval_mode = True)
    
    if args.model_load_dir:
        save_dir = args.model_load_dir
    else:
        save_dir = os.path.join(args.save_dir, f"{args.expr_name}_eval")
        os.makedirs(save_dir, exist_ok=True)
    
    # -- start eval.
    print(f"\n\n{'='*20} Begin Evaluation (max decoding steps = {args.decode_max_length}) {'='*20}\n\n")

    results = {}
    if args.eval:
        # -- Validation
        eval_progress = tqdm(total=len(eval_dataloader), desc="[Eval progress]", leave=True, position=0)
        eval_results = validate(eval_dataloader, model, device, pad_token_id, eos_token_id, eval_progress, args)
        results = {**results, **eval_results}
        eval_progress.close()
    if args.decode and args.bucketing:
        # -- Decoding (only support bucketing batching atm)
        decoding_progress = tqdm(
            total=len(decode_dataloader) if args.decode_steps == -1 else args.decode_steps, 
            desc="[Decoding progress]", 
            leave=True, 
            position=1 if args.eval else 0)
        metrics, decodings = utils.decode(decode_dataloader, model, tokenizer, device, decoding_progress, args)
        decoding_progress.close()
        results = {**results, **metrics}
        pickle.dump(decodings, open(os.path.join(save_dir, "decoding.pkl"), "wb"))
        
    json.dump(results, open(os.path.join(save_dir, "eval_results.json"), "w"))

    print("".format('\n'*7))        
    print(f"Eval results:")
    pprint.pprint(results)
    wandb.log(results)
                   
    
def parse_args(input_args:list):
    parser = argparse.ArgumentParser(description="Evaluation sciprt for GPT-2 models on wikitext-{103,2} dataset.")
    # -- loading & saving args
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--dataset-load-dir", type=str, default=os.path.join(os.environ['STRLM_SRC_DIR'], "training_data"),)
    parser.add_argument("--cache-dir", type=str, default=os.path.join(os.environ.get("STRLM_CHECKPOINT_DIR"), "transformers_cache"))
    parser.add_argument("--save-dir", type=str, default=os.path.join(os.environ["STRLM_CHECKPOINT_DIR"], "gpt2"),)
    parser.add_argument("--expr-name", type=str, required=True)
    
    # -- model & dataset    
    parser.add_argument("--model-name", type=str, choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="gpt2")
    parser.add_argument("--dataset", type=str, choices=["wikitext-103-raw-v1","wikitext-2-raw-v1"], default="wikitext-103-raw-v1")
    parser.add_argument('--loss', choices=['mle', 'st', 'nmst'], default='mle', help="mle: vanila AR / st: monotonic self-terminating / nmst: non-monotonic self-terminating")
    parser.add_argument("--epsilon", type=float, default=0.001, help="A value that control the rate at which p(eos) lowerbound converges to 1.0")
    
    # -- dataset preprocessed format
    parser.add_argument('--sentencized', type=int, choices=[0,1], default=0)
    parser.add_argument('--bucketing', type=int, choices=[0,1], default=1)

    # -- evaluation args
    parser.add_argument('--seed', type=int, default=42, help="A seed number for a reproducible expr.")
    parser.add_argument('--eval', type=int, choices=[0,1], default=1, help="Measure ppl over eval dataset.")
    parser.add_argument('--eval-batch-size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--eval-split', type=str, choices=['validation', 'test'], default='test')
    
    # - decoding args    
    parser.add_argument('--decode', type=int, choices=[0,1], default=0, help="Decode during evalutaion.")
    parser.add_argument('--decode-steps', type=int, default=-1, help="Number of batches to use for every decoding iter. (-1) to iterate over the entire dataset.")
    parser.add_argument('--decode-split', type=str, choices=['validation', 'test'], default='test')
    parser.add_argument('--eval-context-length', type=int, default=10)
    parser.add_argument('--decode-max-length', type=int, default=1000, help="Non-term. ratio threshold.")
    parser.add_argument('--greedy', type=int,  choices=[0,1], default=0, help="Greedy search.")
    parser.add_argument('--sample', type=int,  choices=[0,1], default=0, help="Ancestral sampling.")
    parser.add_argument('--topk', type=int,  choices=[0,1], default=0, help="Top-k sampling.")
    parser.add_argument('--k', type=int, default=2, help="k for top-k sampling.")
    parser.add_argument('--topp', type=int,  choices=[0,1], default=0, help="Nuecleus sampling.")
    parser.add_argument('--p', type=float,  default=0.4, help="p for nuecleus sampling.")
    parser.add_argument('--beam-search', type=int,  choices=[0,1], default=0, help="Beam search.")
    parser.add_argument('--beam-size', type=int, default=2, help="Beam size for beam search.")

    # -- gpu-id for multi-gpu training
    parser.add_argument("--gpu-id", type=int, default=0)

    # -- weights and biases config.
    parser.add_argument("--wandb", type=int, default=0, choices=[0, 1], help="Set 1 to use wandb.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb experiment name.")

    args = parser.parse_args(input_args)
    if args.decode:
        assert args.greedy + args.sample + args.beam_search + args.topk + args.topp == 1, "Only choose one decoding method."
        if args.sample:
            args.decoding_method = "ancestral"
        elif args.beam_search:
            args.decoding_method = "beam_search"
        elif args.topk:
            args.decoding_method = "topk"
        elif args.topp:
            args.decoding_method = "topp"
        else:
            args.decoding_method = "" # I already created a repo for greedy without decoding_method desc.
    if args.topk:
        assert args.k > 0, "k has to be greater than zero."
    elif args.topp:
        0 <= args.p and args.p <= 1, "p has to be in [0,1]."
    elif args.beam_search:
        assert args.beam_size > 0, "Beam size has to be greater than zero."
    
    return args

def main():
    parse_args_and_eval(None)

if __name__ == "__main__":
    main()
