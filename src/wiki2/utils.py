import argparse
import attrdict
import os
import pickle
import random
import torch
import numpy as np
import data
import model_utils

# --- experiment utils
def setup_rnn(
    vocab: data.Dictionary, 
    device: torch.device,
    args: argparse.Namespace,
):
    """A method for setting up a model for training according to expr args.
    """
    args.num_embeddings = len(vocab)
    args.output_dim = len(vocab)
    args.pad_idx = vocab.get_id("<pad>")
    args.eos_idx = vocab.get_id("<eos>")
    args.input_size = args.embedding_dim
    
    if args.loss == "st":
        return setup(model_utils.SelfTerminatingLM, device, args)
    elif args.loss == "nmst":
        return setup(model_utils.NonMonotonicSTLM, device, args)
    else:
        return setup(model_utils.VanillaLM, device, args)
    

def setup(
    model_class: torch.nn.Module, 
    device: torch.device,
    args: argparse.Namespace,
):
    """A helper method for setup_rnn to init model, loss, and optimizer.
    
    :param model_class: a device to train/eval the model on.
    :param device: a device to train/eval the model on.
    :param args: a device to train/eval the model on.
    """
    # model
    model = model_class(args).to(device)
    
    # loss
    if args.loss == "mle":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.pad_idx, reduction="none")
    else:
        criterion = torch.nn.NLLLoss(ignore_index=args.pad_idx, reduction="none")
    
    # optimizer
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model_parameters, lr=args.lr)
    else: # "sgd"
        optimizer = torch.optim.SGD(model_parameters, lr=args.lr, momentum=args.momentum_sgd)

    return model, criterion, optimizer


def save(
    model: torch.nn.Module, 
    vocab: data.Dictionary,
    model_args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    save_dir: str,
    save_name: str,
):
    """A method for saving model checkpoint, model args, model vocab and optimizer state.
    
    :param model: model to save a copy of
    :param vocab: model vocabulary dictionary mapping token to its token_id (and vice-versa).
    :param model_args: expr and model related args. 
    :param optimizer: an optimizer.
    :param scheduler: an scheduler.
    :param save_dir: a directory to save the model checkpoint.
    :param save_name: model checkpoint file name.
    """
    os.makedirs(save_dir, exist_ok = True)
    torch.save({"model_dict": model.state_dict()}, os.path.join(save_dir, f"{save_name}.pt"))
    pickle.dump(vocab, open(os.path.join(save_dir, f"{save_name}_vocab.pkl"), "wb"))
    pickle.dump(model_args, open(os.path.join(save_dir,f"{save_name}_args.pkl"), "wb"))
    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, f"{save_name}_optimizer.pth"))
    print(f"Model saved: {save_name}")


def setup_expr(
    args: argparse.Namespace
):
    """A method for creating model checkpoint saving dir.
    """
    if args.dry_run:
        print(f"\n(!) Dry-run: model/result is not saved.")
    else:
        args.save_dir = _expr_dir(args)
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"\nSave dir: {args.save_dir}")
    set_seed(args.seed)
    return args


def _expr_dir(
    args: argparse.Namespace
):
    """A helper method for `setup_expr` method to avoid overriding existing dir.
    """
    save_dir = os.path.join(args.save_base_dir, args.expr_name)
    i = 2
    save_dir_ = save_dir
    while os.path.exists(save_dir_):
        save_dir_ = os.path.join(save_dir, f"_{i}")
        i += 1
    return save_dir_


def get_model_name(
    args: argparse.Namespace
):
    """A helper method for printing the full name of current model of the expr.
    """
    if args.loss == "nmst": 
        return f"non-monotonic self-terminating language model with ε = {args.epsilon}"
    elif args.loss == "st": 
        return f"monotonic self-terminating language model with ε = {args.epsilon}"
    else: 
        return "regular recurrent language model"

    
def set_seed(
    seed:int
):
    """A method for fixing random seeds for reproducible expr.
    
    :param seed: seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def load_checkpoint(
    load_dir: str, 
    device: torch.device, 
    args: argparse.Namespace
):
    """A method for loading model checkpoint, model args and vocab.

    :param save_dir: a dictionary to load model checkpoint from.
    :param device: device to load the model on (cpu/gpu).
    :param args: expr args.
    """
    model_args = pickle.load(open(os.path.join(load_dir, "model_best_args.pkl"), "rb"))
    vocab = pickle.load(open(os.path.join(load_dir, "model_best_vocab.pkl"), "rb"))
    ckpt = torch.load(os.path.join(load_dir, "model_best.pt"), map_location=torch.device(device))
    args.mask_context_k = model_args.mask_context_k
    args = attrdict.AttrDict({**model_args.__dict__, **args.__dict__})
    
    return vocab, ckpt, args


def load_optims(
    load_dir: str, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
):
    """A method for loading model checkpoint, model args and vocab.

    :param load_dir: a dictionary to load model checkpoint from.
    :param device: device to load the model on (cpu/gpu).
    :param args: expr args.
    """
    checkpoint = torch.load(os.path.join(args.load_dir, "optimizer_model_best.pth"), map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    return optimizer, scheduler
    
