import argparse
import torch
import random
import numpy as np
from st import GPT2STHeadModel
from nmst import GPT2NMSTHeadModel
from transformers import (
    GPT2LMHeadModel,
    get_scheduler,
)
from collections import defaultdict
from metrics import GenerationMetrics

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(args:argparse.Namespace):
    if args.model_load_dir:
        model = _load_model(args.loss, args.model_load_dir)
        model.epsilon = args.epsilon
    else:
        model = _load_model(args.loss, args.model_name)
        model.epsilon = args.epsilon
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*10} {model.__class__.__name__}{'' if args.loss == 'mle' else ' w/ ε = ' + str(args.epsilon)} ({params} params cnt) {'='*10}")
    return model

def _load_model(loss:str, model_:str):
    if loss == "mle":
        model = GPT2LMHeadModel.from_pretrained(model_)
    elif loss == "st":
        model = GPT2STHeadModel.from_pretrained(model_)
    elif loss == "nmst": 
        model = GPT2NMSTHeadModel.from_pretrained(model_)
    else:
        raise NotImplementedError
    return model
    

def load_optimizers(model:torch.nn.Module, args:argparse.Namespace, device:str):
    # Optimizer: Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,},
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon
    )

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    if args.model_load_dir:
        checkpoint = torch.load(os.path.join(args.model_load_dir, "optimizer.pth"), map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    return optimizer, lr_scheduler

def get_hit_rate(lprobs:torch.Tensor, targets:torch.LongTensor, pad:int, hit_thresholds:list = [1,10,20,30]):
    target_lprobs = lprobs.gather(dim=-1, index=targets.where(targets != -100, torch.zeros_like(targets)).unsqueeze(-1))
    pad_mask = targets.ne(-100).float()
    target_rank = (lprobs > target_lprobs).float().sum(dim=-1) # rank from 0 to |V| - 1

    hit_rates = defaultdict(int)
    
    for threshold in hit_thresholds:
        hit_rates[threshold] = ((target_rank < threshold)*pad_mask).sum().item()
    return hit_rates

def decode(eval_dataloader: torch.utils.data.DataLoader, 
           model: torch.nn.Module,
           tokenizer,
           device:str,
           tqdm_bar,
           args: argparse.Namespace):
    model.eval()
    bpe_decoding_continuations = []
    bpe_target_continuations = []
    bpe_decoding_including_prefixes = []
    text_decoding_including_prefixes = []
    text_prefixes = []
    bpe_prefixes = []
        
    gen_metrics = GenerationMetrics()
    decode_steps = args.decode_steps
    context_length = args.eval_context_length
    num_seqs = 0
    
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = batch.squeeze(0)
            num_seqs += batch.size(0)
            
            # skip batches with <PAD> in the context (plus 1 to ensure a non-empty continuation)
            if (batch[:, :context_length+1] == tokenizer.pad_token_id).sum() > 0:
                decode_steps += 1 # ensure we decode `decode_steps` many steps.
                continue
                
            bpe_prefix, text_prefix, bpes, texts, outputs = generate_batch(
                model, tokenizer, batch, context_length, device,
                max_length=args.decode_max_length, args=args,
            )

            bpes_ = [b[args.eval_context_length:] for b in bpes]  # original has prefix

            prefix_, target_trim, model_trim = trim(
                bpes_, batch, context_length, tokenizer.eos_token_id
            )

            gen_metrics.step(
                model_trim, target_trim, outputs, tokenizer.eos_token_id, context_length
            )

            bpe_decoding_continuations.extend(model_trim)
            bpe_decoding_including_prefixes.extend(bpes)
            text_decoding_including_prefixes.extend(texts)

            text_prefixes.extend(text_prefix)
            bpe_prefixes.extend(bpe_prefix)
            
            # print(texts)
            
            bpe_target_continuations.extend(target_trim)

            tqdm_bar.desc = "[Decoding progress] Avg. len: %.2f, Non-term: %.2f" % (
                np.mean([len(b) for b in bpe_decoding_including_prefixes]),
                np.mean([len(b) == args.decode_max_length for b in bpe_decoding_including_prefixes])
            )
            tqdm_bar.update()
            step += 1
            if decode_steps > 0 and step == decode_steps:
                break

    decodings = {
        'text_decoding_including_prefix': text_decoding_including_prefixes,
        'bpe_decoding_continuation': bpe_decoding_continuations,
        'text_prefix': text_prefixes,
        'bpe_prefix': bpe_prefixes,
        'bpe_target_continuation': bpe_target_continuations
    }

    metrics = {}
    gen_metrics = gen_metrics.normalize('valid/decode')
    for k, v in gen_metrics.items():
        metrics[k] = v
    metrics["valid/decode/num_seqs"] = num_seqs
    return metrics, decodings


def generate_batch(model, tokenizer, batch, context_length, device, max_length, args,):
    with torch.no_grad():
        batch, max_len_in_batch = wrap_context_batch(batch, context_length)
        attention_mask=batch.ne(tokenizer.pad_token_id).float()
        batch = batch.to(device)
        attention_mask = attention_mask.to(device)
        bpe_prefixes = batch.tolist()
        text_prefixes = [tokenizer.decode(p) for p in bpe_prefixes]
        bpe_decodings = []
        text_decodings = []

        if batch.size(0) > 0:
            if args.greedy:
                outputs = model.generate(
                    batch,
                    max_length=max_length,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=attention_mask,
                )
            elif args.sample:
                outputs = model.generate(
                    batch,
                    max_length=max_length,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=attention_mask,
                )
            elif args.topk:
                outputs = model.generate(
                    batch,
                    max_length=max_length,
                    do_sample=True,
                    top_k=args.k,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=attention_mask,
                )
            elif args.topp:
                outputs = model.generate(
                    batch,
                    max_length=max_length,
                    do_sample=True,
                    top_p=args.p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=attention_mask,
                )
            elif args.beam_search:
                outputs = model.generate(
                    batch,
                    max_length=max_length,
                    do_sample=False,
                    num_beams=args.beam_size,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=attention_mask,
                )
            full_outputs = outputs.clone()
            for b_ind in range(outputs.size(0)):
                bpe_decoding = outputs[b_ind].tolist()
                if tokenizer.eos_token_id in bpe_decoding:
                    bpe_decoding = bpe_decoding[:bpe_decoding.index(tokenizer.eos_token_id)+1]
                text_decoding = tokenizer.decode(bpe_decoding)

                bpe_decodings.append(bpe_decoding)
                text_decodings.append(text_decoding)

    return bpe_prefixes, text_prefixes, bpe_decodings, text_decodings, full_outputs


def trim(decoded, target, context_length, eos_id):
    prefix = target[:, :context_length].tolist()
    cont_target = target[:, context_length:]
    target_trim = []
    model_trim = []
    if not isinstance(decoded, list):
        decoded = decoded.tolist()
    if not isinstance(cont_target, list):
        cont_target = cont_target.tolist()

    for data_cont, model_cont in zip(cont_target, decoded):
        if eos_id in data_cont:
            data_cont_ = data_cont[:data_cont.index(eos_id)+1]
        else:
            data_cont_ = data_cont
        if eos_id in model_cont:
            model_cont_= model_cont[:model_cont.index(eos_id)+1]
        else:
            model_cont_ = model_cont
        target_trim.append(data_cont_)
        model_trim.append(model_cont_)
    return prefix, target_trim, model_trim


def wrap_context_batch(batch, context_length):
    max_len_in_batch = max([i.numel() for i in batch])
    context_list = []
    for seq in batch:
        if seq.size(0) < context_length:
            continue
        else:
            context_list.append(seq[:context_length])
    if len(context_list) == 0:
        return torch.tensor([], dtype=torch.long), max_len_in_batch
    else:
        return torch.stack(context_list, dim=0), max_len_in_batch
