import argparse
import pickle
import os
import tqdm
import torch
import wandb
import pprint
import utils
import data
import decoding_utils
from evaluate import evaluate

def train(
    model: torch.nn.Module, 
    vocab: data.Dictionary,
    data_loaders: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    criterion: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
):
    """Training method.
    
    :param model: Model for training.
    :param vocab: A dictionary lookup module mapping a token to its token_id (and vice-versa).
    :param data_loaders: A dictionary of dataloader with its keys denoting the dataset split and its values
    :param optimizer: an optimizer for training.
    :param scheduler: a learning rate scheduler.
    :param criterion: a loss function (torch.nn.CrossEntropyLoss for 'mle' or torch.nn.NLLLoss for 'nmst' and 'st').
    :param device: device to run the computation on (cpu/gpu).
    :param args: expr args.
    """
    best_val_loss = 1e5
    early_stop = 0
    
    eos_idx = vocab.get_id("<eos>")
    pad_idx = vocab.get_id("<pad>")
    cpad_idx = vocab.get_id("<cpad>")
    vocab_size = len(vocab)
    
    for epoch_number in range(args.max_epochs):
        
        # -- training
        print(f"Epoch {epoch_number}:\n")
        
        sum_num_pred_tokens = 0
        sum_loss = 0.0
        
        model.train()
        
        tqdm_prefix = tqdm.utils._term_move_up() + '\r'
        pbar = tqdm.auto.tqdm(enumerate(data_loaders["train"]), total = len(data_loaders["train"]), desc=f"Train")
        
        for i, (inp, target) in pbar:
            optimizer.zero_grad()
            
            inp = inp.to(device)
            target = target.to(device)
            
            num_pred_tokens = target.ne(pad_idx).count_nonzero().item()
            num_pred_tokens -= target.eq(cpad_idx).count_nonzero().item()
            sum_num_pred_tokens += num_pred_tokens
            
            output = model(inp)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1)).sum()
            sum_loss += loss.item()
            
            loss = loss / num_pred_tokens # average the current loss.
            loss.backward()
            
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            optimizer.step()
            
            avg_loss = sum_loss / sum_num_pred_tokens
            wandb.log({"train/loss":avg_loss, "train_inner/lr": scheduler.get_last_lr()[0],})
            tqdm.auto.tqdm.write(tqdm_prefix + f"[Step: {i:4d}]  NLL Loss: {avg_loss:6.4f}  ")

        # -- validation
        sum_num_pred_tokens = 0
        sum_loss = 0.0
        
        model.eval()
        pbar = tqdm.auto.tqdm(enumerate(data_loaders["validation"]), total = len(data_loaders["validation"]), desc=f"Valid")
        with torch.no_grad():
            for i, (inp, target) in pbar:
                
                inp = inp.to(device)
                target = target.to(device)
                
                output = model(inp)

                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                    
                # mask loss computed over the context C
                if args.mask_context_k > 0:
                    loss = loss.view(target.size())
                    loss[:,:args.mask_context_k] = 0.0
                        
                num_context_tokens = target.size(0) * args.mask_context_k
                num_pred_tokens = target.ne(pad_idx).count_nonzero().item() - num_context_tokens
                sum_num_pred_tokens += num_pred_tokens

                sum_loss += loss.sum()
                        
            avg_loss = sum_loss / sum_num_pred_tokens
            
            # -- get non-termination stats using a fraction of samples from validation set
            decoding_stats = decoding_utils.decoding_dataset_stats(
                model,
                {'validation': data_loaders['validation']},
                vocab,
                device,
                num_samples={'validation': args.num_samples},
                max_steps=args.max_sample_steps,
                prefix_length=args.mask_context_k,
                decoding=("greedy",),
                consistent_sampling=False,
            )
            
            
            if avg_loss < best_val_loss:
                if not args.dry_run:
                    utils.save(
                        model = model,
                        vocab = vocab,
                        model_args = args,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        save_dir = args.save_dir,
                        save_name = "model_best",
                    )
                early_stop = 0
                best_val_loss = avg_loss
            else:
                early_stop += 1
                scheduler.step()
                
            ppl = avg_loss.exp().item()
            best_ppl = best_val_loss.exp().item() if epoch_number > 0 else ppl
            # ppl = utils.perplexity(avg_loss.item())
            # best_ppl = utils.perplexity(best_val_loss.item()) if epoch_number > 0 else ppl
           
            # -- print val. results
            print(f"Epoch {epoch_number} complete.\t Val loss {avg_loss.item():.4f}\t PPL {ppl:.2f} (best {best_ppl:.2f})")
            for dataset_split in decoding_stats:
                decoding_stats_ = decoding_stats[dataset_split]
                print((f"{dataset_split}: "
                       f"non-term greedy {decoding_stats_['greedy']['nonterminated']:.4E} "
                       f"(avg. len {decoding_stats_['greedy']['avg_len']:.1f}, "
                       f"uniq. {decoding_stats_['greedy']['uniq_nonterminated']:.4f})\n"))
        
        log_dict = {"val/avg_val_loss":avg_loss.item(), "val/ppl":ppl, "val/best_ppl": best_ppl, "val/early_stop": early_stop}       
        for key in list(decoding_stats.keys()):
            log_dict[f"val/decoding/{key}"] = decoding_stats.pop(key)
        wandb.log(log_dict)
        
        # -- stop training if val. loss didn't improve for the last args.early_stop many epochs.
        if early_stop >= args.early_stop:
            break
    
    # -- final eval.
    if not args.dry_run:
        print("Performing final evaluation...")
        ckpt = torch.load(os.path.join(args.save_dir, "model_best.pt"), map_location=torch.device(device))
        model.load_state_dict(ckpt["model_dict"])
        args.eval = 1
        args.decode = 1
        args.decoding_log_name = "decoding_log_greedy.pkl"

        ppl, decoding_stats = evaluate(
            output_dir = args.save_dir,
            model = model,
            data_loaders = data_loaders,
            vocab = vocab,
            device = device,
            criterion = criterion,
            args = args,
        )
        wandb.log(ppl)
        wandb.log(decoding_stats)

    
def main(args):
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")  
    
    data_loaders, vocab = data.get_dataloaders_and_vocab(args)
    
    # -- for finetuning
    if args.model_load_dir:
        print(f"Loading the model from {args.model_load_dir}...")
        vocab, ckpt, args = utils.load_checkpoint(args.model_load_dir, device, args)
        model, criterion, optimizer = utils.setup_rnn(vocab, device, args)
        model.load_state_dict(ckpt["model_dict"])
    else:
        model, criterion, optimizer = utils.setup_rnn(vocab, device, args)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_anneal)
    
    # -- for finetuning
    if args.model_load_dir and args.load_optimizer:
        optimizer, scheduler = utils.load_optims(args.model_load_dir, optimizer, scheduler)
    
    # -- initialize weights and biases
    wandb.init(
        project="nmstlm-wiki2",
        config=args,
        job_type="train",
        name=args.expr_name,
        mode="online" if args.wandb == 1 else "disabled",
    )
    
    # -- display (1) expr name, (2) model arch, (3) optim setting, and (4) expr args.
    print(f"\nExperiment name: {args.expr_name} (seed = {args.seed})") # (1)
    print(f'Model: {utils.get_model_name(args)}') # (2)
    print(f"params: {sum(p.numel() for p in model.parameters())}")
    print(f'{model}\n')
    print(f'{optimizer}\n') # (3)
    print(f"{pprint.pformat(dict(args) if args.model_load_dir else vars(args))}\n") # (4)
    
    # -- train!
    train(model, vocab, data_loaders, optimizer, scheduler, criterion, device, args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for WikiText-2 expr in 'A Non-monotoic Self-terminating Language Model' (2022).")
    
    # -- model checkpoint loading/saving args.
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--load-optimizer", type=int, default=0, choices=[0, 1], 
                        help="Loads optim and scheduler for an extended training using a checkpoint")
    parser.add_argument("--save-base-dir", type=str, default=os.path.join(os.environ["NMST_DIR"], "checkpoint/wiki2"))
    
    # -- dataset loading params.
    parser.add_argument("--dataset-load-dir", type=str, default=os.path.join(os.environ["NMST_DIR"],"data"))
    parser.add_argument("--dataset-version", type=str, choices=["wikitext-2-v1", "wikitext-2-v1-raw"], default="wikitext-2-v1")
    parser.add_argument("--line-by-line", type=int, choices=[0, 1], default=0, help="Arg. for segementing the dataset into sentences.",)
    
    parser.add_argument("--expr-name", type=str, required=True)
    parser.add_argument("--dry-run", type=int, default=0, choices=[0, 1], help="Skip model saving for debugging.")
    
    # TRAINING RELATED PARAMS:
    # -- model params.
    parser.add_argument("--rnn-type", type=str, default="nn.LSTM", choices=["nn.RNN", "nn.LSTM", "nn.GRU"])
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rnn-dropout", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--tie-weights", type=int, default=1, choices=[0, 1])
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    
    parser.add_argument(
        "--loss", type=str, choices=["nmst", "st", "mle"], default="nmst", 
        help=(
            "'nmst': Non-monotoic self-terminating LM || "
            "'st': (monotoic) self-terminating LM || "
            "'mle': vanilla LM"
        )
    )
    parser.add_argument("--epsilon", type=float, default=0.0001, help="Epsilon value for 'nmst' and 'st' models.")
    
    # -- optimizer related params.
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--momentum-sgd", type=float, default=0.99)
    parser.add_argument("--lr-anneal", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # experiment hyperparams:
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=70)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # -- context masking
    parser.add_argument("--mask-context-k", type=int, default=10, help="If > 0, (val) loss of prefix up to k is masked.")
    
    # -- validation phase decoding
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-random-prefixes", type=int, default=1000)
    parser.add_argument("--max-sample-steps", type=int, default=1500)
    parser.add_argument("--decode", type=int, default=1, choices=[0,1])
    parser.add_argument("--greedy", type=int, default=1, choices=[0,1])
    parser.add_argument("--num-train-samples", type=int, default=-1)

    # -- for enabling weights and biases
    parser.add_argument("--wandb", type=int, default=1, choices=[0, 1], help="Set 1 to use wandb.")
    parser.add_argument("--log-every", type=int, default=100, help="Wandb training log interval (in steps).")
    
    args = utils.setup_expr(parser.parse_args())
    main(args)
