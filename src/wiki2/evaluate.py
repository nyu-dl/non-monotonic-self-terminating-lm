import argparse
import pickle
import json
import os
import torch
import pprint
from natsort import realsorted
from glob import glob
import utils
import data
import decoding_utils


def main(args: argparse.Namespace):
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.sweep_dir:
        dirs = realsorted(glob(os.path.join(args.sweep_dir, "*")))
    elif args.model_load_dir:
        dirs = [args.model_load_dir]
    else:
        raise ValueError("Provide either 'sweep_dir' or 'model_load_dir' for evaluation.")
    
    print(f"\n{'='*10} Evaluating best model checkpoint(s) in the folowing dirs {'='*10}")
    print("\n".join(dirs))
    
    args_dict = args.__dict__
    
    for d in dirs:
        print(f"\n{'='*10} Evaluating {d} {'='*10}")
        # -- load model and args
        vocab, ckpt, args = utils.load_checkpoint(d, device, args)
        model, criterion, _ = utils.setup_rnn(vocab, device, args)
        model.load_state_dict(ckpt["model_dict"])
        data_loaders, _ = data.get_dataloaders_and_vocab(args)
        
        output_dir = os.path.join(d, "eval")
        os.makedirs(output_dir, exist_ok=True)
        if args.decoding_log_name is None:
            if args.greedy:
                args.decoding_log_name = f"decoding_log_greedy.pkl"
            elif args.beam_search:
                args.decoding_log_name = f"decoding_log_beam_search_{args.beam_size}.pkl"
            elif args.sample:
                args.decoding_log_name = f"decoding_log_sample.pkl"
            elif args.topk:
                args.decoding_log_name = f"decoding_log_topk_{args.k}.pkl"
            elif args.topp:
                args.decoding_log_name = f"decoding_log_topp_{args.p}.pkl"
            elif args.consistent_sampling:
                args.decoding_log_name = f"decoding_log_consistent_{args.max_sample_steps}.pkl"
        if args.eval_log_name is None:
            args.eval_log_name = f"eval_log_{args.max_sample_steps}.json"

        for ds in args.exclude_datasets:
            try:
                del data_loaders[ds]
                print(f'{ds} part is excluded from eval')
            except:
                pass
                    
        print(model)
        print(f"{pprint.pformat(dict(args) if args.model_load_dir else vars(args))}\n")
        evaluate(
            output_dir = output_dir,
            model = model,
            data_loaders = data_loaders,
            vocab = vocab,
            device = device,
            criterion = criterion,
            args = args,
        )


def evaluate(
    output_dir: str,
    model: torch.nn.Module,
    data_loaders: dict,
    vocab: data.Dictionary,
    device: torch.device,
    criterion: torch.nn.Module,
    args: argparse.Namespace,
):
    """A method for model evaluation (ppl measurement and decoding).
    
    :param output_dir: A directory to store the evaluation outputs.
    :param model: A model to evaluate.
    :param data_loaders: A dictionary of dataloader with its keys denoting the dataset split and its values
    :param vocab: A dictionary lookup module mapping a token to its token_id (and vice-versa).
    :param device: Device to run the computation on (cpu/gpu).
    :param criterion: A loss function (either torch.nn.CrossEntropyLoss or torch.nn.NLLLoss).
    :param args: Evaluation args.
    """
    model.eval()
    if args.eval:
        ppls = {}
        for data_mode, data_loader_ in data_loaders.items():
            if data_mode == 'random':
                # no targets and ppl for random
                continue
            print(f'Computing PPL for {data_mode}')
            logs = decoding_utils.compute_ppl_dataloader(
                args,
                vocab,
                model,
                criterion,
                data_loader_,
                data_mode,
                device,
            )
            ppls[data_mode] = logs
        print(pprint.pformat(ppls))
        if hasattr(args, 'eval_log_name') == False:
            args.eval_log_name = f"eval_log_{args.max_sample_steps}.json"
        json.dump(ppls, open(os.path.join(output_dir, args.eval_log_name), "w"))
    if args.decode:
        if args.greedy:
            decoding_algos = ["greedy"]
        elif args.beam_search:
            decoding_algos = [f"beam_{args.beam_size}"]
        elif args.sample:
            decoding_algos = ["sample"]
        elif args.topk:
            decoding_algos = [("topk", args.k)]
        elif args.topp:
            decoding_algos = [("topp", args.p)]
        elif args.consistent_sampling:
            decoding_algos = (("topk", 2), ("topk", 4), ("topp", 0.2),("topp", 0.4))
        decoding_stats = decoding_utils.decoding_dataset_stats(
            model,
            data_loaders,
            vocab,
            device,
            num_samples={'train': args.num_train_samples},
            max_steps=args.max_sample_steps, #model_args.max_sample_steps,
            temperature=1.0,
            prefix_length=args.mask_context_k,
            decoding=decoding_algos,
            consistent_sampling=args.consistent_sampling if hasattr(args, 'consistent_sampling') else 0,
            save_decoding_logs=args.save_decoding_logs if hasattr(args, 'save_decoding_logs') else 0,
        )
        pickle.dump(decoding_stats, open(os.path.join(output_dir, args.decoding_log_name), "wb"))
        
    if args.decode and args.eval:
        return ppls, decoding_stats
    elif args.decode:
        return decoding_stats
    elif args.eval:
        return ppls



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for WikiText-2 expr in 'A Non-monotoic Self-terminating Language Model' (2022).")
    parser.add_argument("--sweep-dir", type=str, default=None, help="A dir containing multiple model checkpoints as subdirs.")
    parser.add_argument("--model-load-dir", type=str, default=None, help="A dir containing a single model checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--exclude-datasets", default=["train","test","random"], type=str, nargs='+')
    
    parser.add_argument("--num-random-prefixes", type=int, default=1000)
    parser.add_argument("--num-train-samples", type=int, default=-1)
    parser.add_argument("--max-sample-steps", type=int, default=70000)

    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--eval", type=int, default=1, choices=[0,1])    
    parser.add_argument("--decode", type=int, default=1, choices=[0,1])
    parser.add_argument("--greedy", type=int, default=0, choices=[0,1])
    parser.add_argument("--beam-search", type=int, default=0, choices=[0,1])
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--sample", type=int, default=0, choices=[0,1])
    parser.add_argument("--topk", type=int, default=0, choices=[0,1])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--topp", type=int, default=0, choices=[0,1])
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--consistent-sampling", type=int, default=0, choices=[0,1])
    
    parser.add_argument("--save-decoding-logs", type=int, default=1, choices=[0,1])
    parser.add_argument("--decoding-log-name", type=str, default=None)
    parser.add_argument("--eval-log-name", type=str, default=None)

    args = parser.parse_args()
    assert args.greedy + args.beam_search + args.sample + args.topk + args.topp + args.consistent_sampling == 1, "Choose one decoding only."
    
    utils.set_seed(args.seed)
    main(args)
