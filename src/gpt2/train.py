import argparse
import os
import torch
import torch.nn.functional as F
import pprint
import wandb
import pickle
from transformers import GPT2Tokenizer
from tqdm import tqdm
from tqdm.utils import _term_move_up
from collections import defaultdict
import data as data
import utils as utils

# -- train + val
def parse_args_and_train(input_args):
    args = parse_args(input_args)
    print(f"\n{'='*20} Training args: {'='*20}")
    print(pprint.pformat(vars(args)))
    
    # -- initialize weights and biases for expr tracking
    wandb.init(
        project="strlm-gpt2",
        config=args,
        job_type="fine-tuning",
        tags="baseline",
        name=args.expr_name,
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
    train_dataloader, eval_dataloader, decode_dataloader = data.load_dataloaders(args, model.config, pad_token_id)
    
    # -- load optimizer & lr scheduler
    optimizer, lr_scheduler = utils.load_optimizers(model, args, device)
        
    # -- start training
    print(f"\n\n{'='*20} Begin Training (max steps = {args.max_train_steps}) {'='*20}\n\n")
    
    training_progress = tqdm(total=args.eval_every if args.eval else args.max_train_steps, desc="[Training progress]", leave=True, position=0)
    
    def print_train_progress(epoch:int, step:int, nll_loss:float, ppl:float, total_tokens:int, hit_rates:dict, learning_rate:float):
        hit_rate_outputs = ""
        total_tokens = float(total_tokens)
        for k,v in hit_rates.items():
            hit_rate_outputs += f"({k}: {v / total_tokens:.4f})  "
        progress = "[Epcoh:{:4}  ||  Step:{:5} / {}]  NLL: {:6.3f}  ||  PPL: {:7.3f}  ||  Hits@: {}||  LR: {:.7e} {}".format(
            epoch, step, args.max_train_steps, nll_loss, ppl, hit_rate_outputs, learning_rate, ' ' * 10
        )
        return progress
    
    epoch = 0
    best_ppl = 1e5
    patience = 0
    total_loss = 0
    total_tokens = 0
    hits_at = defaultdict(int)
    tqdm_p = _term_move_up() + '\r'
    
    model.train()

    for step, batch in zip(range(args.max_train_steps), data.InfiniteYield(train_dataloader)):
        optimizer.zero_grad()
        
        # batch training with adaptive bs (for minimal padding) -> [1,B,T].
        if args.bucketing:
            batch = batch.squeeze(0)
            attention_mask = batch.ne(pad_token_id).float()
            labels = batch.where(batch != pad_token_id, torch.full_like(batch, -100))
            batch[batch == pad_token_id] = 0 # PAD is a non-registered token (non-registered token-id will trigger CUDA DEVICE ERR).
            # vanila lm.
            if args.loss == "mle":
                output = model(
                    input_ids = batch.to(device),
                    attention_mask = attention_mask.to(device),
                    labels = labels.to(device),
                )
                loss = output.loss
                hit_rates = utils.get_hit_rate(output.logits[:,:-1,:], labels[:,1:].to(device), pad_token_id)
            # st and nmst.
            else:
                output = model(
                    input_ids = batch.to(device),
                    attention_mask = attention_mask.to(device),
                )
                # st/nmst lprobs.
                lprobs = model.st_softmax(
                    output.logits, # vanila lm logits.
                    eos_token_id,
                )
                loss = F.nll_loss(
                    lprobs[:,:-1,:].reshape(-1, lprobs.size(-1)),
                    labels[:,1:].to(device).reshape(-1),
                )
                hit_rates = utils.get_hit_rate(lprobs[:,:-1,:], labels[:,1:].to(device), pad_token_id)
            n_tokens = attention_mask.count_nonzero()
            
        # batch training with a constant bs [B,T].
        else:
            batch["input_ids"][batch["input_ids"] == pad_token_id] = 0 
            output = model(
                input_ids = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels = batch["labels"].to(device),
            )
            loss = output.loss
            n_tokens = batch["attention_mask"].count_nonzero()
            hit_rates = utils.get_hit_rate(output.logits[:,:-1,:], batch["labels"][:,1:].to(device), pad_token_id)
        for k,v in hit_rates.items():
            hits_at[k] += v
        
        total_loss = total_loss + (loss * n_tokens)
        total_tokens = total_tokens + n_tokens

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        cur_loss = (total_loss / total_tokens)
        
        # progress bar update
        training_progress.update()
        tqdm.write(tqdm_p + print_train_progress(epoch, step, cur_loss.item(), cur_loss.exp().item(), total_tokens, hits_at, lr_scheduler.get_last_lr()[0]))
        
        # wandb logging
        train_inner = {
            "train_inner/nll_loss": cur_loss.item(),
            "train_inner/ppl": cur_loss.exp().item(),
            "train_inner/lr": lr_scheduler.get_last_lr()[0], # lr for groups w/ weight decay.
        }
        for k,v in hits_at.items():
            train_inner[f"train_inner/hits_at_{k}"] = float(v) / total_tokens.item()
        wandb.log(train_inner)
        
        epoch = step // len(train_dataloader)
        
        if args.eval and step % args.eval_every == 0:
            # -- Validation
            validation_progress = tqdm(total=len(eval_dataloader), desc="[Validation progress]", leave=False, position=1)
            val_results = validate(eval_dataloader, model, device, pad_token_id, eos_token_id, validation_progress, args)
            validation_progress.close()
            if args.decode and args.bucketing:
                # -- Decoding (only support bucketing batching atm)
                decoding_progress = tqdm(total=args.decode_steps, desc="[Decoding progress]", leave=False, position=2)
                metrics, decodings = utils.decode(decode_dataloader, model, tokenizer, device, decoding_progress, args)
                decoding_progress.close()
            val_ppl = val_results["valid/ppl"]
            print("".format('\n'*4))
            if val_ppl < best_ppl:
                if not args.dry_run:
                    save_dir = os.path.join(args.save_dir, f"{args.expr_name}_{args.seed}_best_run")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    checkpoint = {
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(save_dir, f"optimizer.pth"))
                    if args.decode:
                        pickle.dump(decodings, open(os.path.join(save_dir, "decoding.pkl"), "wb"))
                    print(f"checkpoint at step {step} saved at {save_dir}")
                best_ppl = val_ppl
                patience = 0
            else:
                patience += 1
            print(f"patience = {patience}\n")
            if args.decode:
                val_results = {**val_results, **metrics}
            val_results["valid/epoch"] = epoch
            val_results["valid/patience"] = patience
            print(f"Validation results at step = {step} / {args.max_train_steps}")
            pprint.pprint(val_results)
            print("\n\n")
            wandb.log(val_results)
            if args.early_stop <= patience:
                print(f"\nearly stopping...")
                break
                
            total_loss = 0
            total_tokens = 0
            hits_at.clear()
            training_progress.reset()
                    

def validate(eval_dataloader, model, device, pad_token_id, eos_idx, tqdm_bar, args):
    model.eval()
    # -- Next-token prediction task (MLE loss)
    total_tokens = 0
    total_loss = 0
    hits_at = defaultdict(int)
    cur_loss = 0
    
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if args.bucketing:
                batch = batch.squeeze(0)
                attention_mask = batch.ne(pad_token_id).float()
                labels = batch.where(batch != pad_token_id, torch.full_like(batch, -100))
                batch[batch == pad_token_id] = 0 
                if args.loss == "mle":
                    output = model(
                        input_ids = batch.to(device),
                        attention_mask = attention_mask.to(device),
                        labels = labels.to(device),
                    )
                    loss = output.loss
                    hit_rates = utils.get_hit_rate(output.logits[:,:-1,:], labels[:,1:].to(device), pad_token_id)
                else:
                    output = model(
                        input_ids = batch.to(device),
                        attention_mask = attention_mask.to(device),
                    )
                    lprobs = model.st_softmax(
                        output.logits,
                        eos_idx,
                    )
                    loss = F.nll_loss(
                        lprobs[:,:-1,:].reshape(-1, lprobs.size(-1)),
                        labels[:,1:].to(device).reshape(-1),
                    )
                    hit_rates = utils.get_hit_rate(lprobs[:,:-1,:], labels[:,1:].to(device), pad_token_id)
                n_tokens = attention_mask.count_nonzero()
            else:
                batch["input_ids"][batch["input_ids"] == pad_token_id] = 0 
                output = model(
                    input_ids = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    labels = batch["labels"].to(device),
                )
                loss = output.loss
                n_tokens = batch["attention_mask"].count_nonzero()
                hit_rates = utils.get_hit_rate(output.logits[:,:-1,:], batch["labels"].to(device), pad_token_id)
            
            for k,v in hit_rates.items():
                hits_at[k] += v
            total_loss = total_loss + (loss * n_tokens)
            total_tokens = total_tokens + n_tokens
            
            cur_loss = (total_loss / total_tokens)
            
            tqdm_bar.desc = "[Validation progress] NLL: {:.2f}, PPL: {:.2f}, Hits: 1 {:.3f} | 10 {:.3f}".format(
                cur_loss.item(), cur_loss.exp().item(), hits_at[1] / total_tokens, hits_at[10] / total_tokens,
            )
            tqdm_bar.update()
        cur_loss = (total_loss / total_tokens)
        val_results = {
            "valid/nll_loss": cur_loss.item(),
            "valid/ppl": cur_loss.exp().item(),
        }
        for k,v in hits_at.items():
            val_results[f"valid/hits_at_{k}"] = v / total_tokens.item()
    return val_results
    
    
def parse_args(input_args:list):
    parser = argparse.ArgumentParser(description="Training sciprt for GPT-2 models on wikitext-[103,2] dataset.")
    # -- loading & saving args
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--dataset-load-dir", type=str, default=os.path.join(os.environ['NMST_DIR'], "data"),)
    parser.add_argument("--cache-dir", type=str, default=os.path.join(os.environ.get("NMST_DIR"), "transformers_cache"))
    parser.add_argument("--save-dir", type=str, default=os.path.join(os.environ["NMST_DIR"], "checkpoint/gpt2"),)
    parser.add_argument("--save-every", type=int, default=500, help="Checkpoint saving interval (in steps).")
    parser.add_argument("--dry-run", type=int, default=0, choices=[0,1], help="Skip model checkpoint saving (for debugging).")
    parser.add_argument("--expr-name", type=str, required=True)
    
    # -- model & dataset    
    parser.add_argument("--model-name", type=str, choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="gpt2")
    parser.add_argument("--dataset", type=str, choices=["wikitext-103-raw-v1","wikitext-2-raw-v1"], default="wikitext-103-raw-v1")
    parser.add_argument('--loss', choices=['mle', 'st', 'nmst'], default='mle', help="mle: vanila AR / st: monotonic self-terminating / nmst: non-monotonic self-terminating")
    parser.add_argument("--epsilon", type=float, default=0.001, help="A value that control the rate at which p(eos) lowerbound converges to 1.0")
    
    # -- dataset preprocessed format
    parser.add_argument('--sentencized', type=int, choices=[0,1], default=0)
    parser.add_argument('--bucketing', type=int, choices=[0,1], default=1)
    
    # -- optimizer (supports AdamW only)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler-type", type=str, default="linear", help="Types of available learning rate schedulers.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num-warmup-steps", type=int, default=0, # default=1000,
        help="Number of steps for the warmup in the lr scheduler."
    )
    
    # -- standard training args
    # - training args
    parser.add_argument('--seed', type=int, default=42, help="A seed number for a reproducible expr.")
    parser.add_argument("--max-train-steps", type=int, default=10000000, help="Total number of training steps to perform.")
    parser.add_argument('--train-batch-size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--context-length', type=int, default=10)
    # - validation args
    parser.add_argument('--eval', type=int, choices=[0,1], default=1, help="Eval while training.")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument('--eval-batch-size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--eval-split', type=str, choices=['validation', 'test'], default='test')
    parser.add_argument('--early-stop', type=int, default=10000)
    # - decoding args    
    parser.add_argument('--decode', type=int, choices=[0,1], default=0, help="Decode during evalutaion.")
    parser.add_argument('--decode-steps', type=int, default=15, help="Number of batches to use for every decoding iter.")
    parser.add_argument('--decode-split', type=str, choices=['validation', 'test'], default='test')
    parser.add_argument('--eval-context-length', type=int, default=10)
    parser.add_argument('--decode-max-length', type=int, default=1000, help="Non-term. ratio threshold.")
    parser.add_argument('--greedy', type=int,  choices=[0,1], default=1, help="Greedy search.")
    parser.add_argument('--sample', type=int,  choices=[0,1], default=0, help="Ancestral sampling.")
    parser.add_argument('--topk', type=int,  choices=[0,1], default=0, help="Top-k sampling.")
    parser.add_argument('--k', type=int, default=1, help="k for top-k sampling.")
    parser.add_argument('--topp', type=int,  choices=[0,1], default=0, help="Nuecleus sampling.")
    parser.add_argument('--p', type=float,  default=1, help="p for nuecleus sampling.")
    parser.add_argument('--beam-search', type=int,  choices=[0,1], default=0, help="Beam search.")
    parser.add_argument('--beam-size', type=int, default=4, help="Beam size for beam search.")

    # -- gpu-id for multi-gpu training
    parser.add_argument("--gpu-id", type=int, default=0)

    # -- weights and biases config.
    parser.add_argument("--wandb", type=int, default=0, choices=[0, 1], help="Set 1 to use wandb.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb experiment name.")

    args = parser.parse_args(input_args)
    return args

def main():
    parse_args_and_train(None)

if __name__ == "__main__":
    main()


