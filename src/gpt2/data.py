import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, default_data_collator
from datasets import load_from_disk
import os
import itertools
from typing import Iterable, Iterator

# -- batch training utils: auto-reload dataloader.
class InfiniteYield(Iterator):
    def __init__(self, iterable: Iterable):
        self.iterable = iterable
        self.iterator = iter(itertools.cycle(self.iterable))

    def __next__(self):
        return next(self.iterator)

    def pop(self):
        return next(self.iterator)
    
# -- (bucketing for seq2seq) stratify input sequences into groups that have roughly the same size and pad them accordingly for minimal padding.
class BucketingDataset(Dataset):
    def __init__(self, data:list, split:str, context_length:int, model_config:GPT2Config, pad_token_id:int):
        self.split = split
        self.total_tokens = 1024
        self.token_limit = model_config.n_positions
        self.pad_token_id = pad_token_id
        self.context_length = context_length
        
        original_seq_cnt = len(data)
        # NOTE: discard sequences shorter than the context size plus 1.
        data = filter(lambda x: len(x) > context_length, data)
        
        # NOTE: discard sequences greater than the maximum sequence length of the model.
        data = list(filter(lambda x: len(x) <= self.token_limit, data))

        self.batches = self._make_batches(data)
        print(f"{'='*10} {split} size: {len(data)} ({original_seq_cnt - len(data)} discarded) (max len %d) (%d batches) {'='*10}" % (
            max(len(d) for d in data),
            len(self.batches)
        ))

    def _make_batches(self, data):
        """Group by similar lengths, then create padded batches that meet the token limit."""
        sorted_data = sorted(data, key=lambda x: -len(x))
        batches = []

        i = 0
        while i < len(sorted_data):
            example = sorted_data[i]

            # The first element will be the longest, which will determine the padded size.
            element_size = len(example)
            batch_size = max(1, self.total_tokens // element_size)

            batch = sorted_data[i:i+batch_size]
            batch = self._pad_batch(batch, element_size)

            batches.append(batch)
            i = i + batch_size

        return batches

    def _pad_batch(self, batch, element_size):
        batch_ = []
        for element in batch:
            element_ = element + [self.pad_token_id]*(element_size - len(element))
            assert len(element_) == element_size
            batch_.append(element_)
        return batch_

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index], dtype=torch.long)


def load_dataloaders(args, model_config, pad_token_id:int, eval_mode:bool = False):
    # -- load dataset
    if args.sentencized:
        # the dataset preprocessed into sentences.
        dataset_file_name = f"{args.dataset}-sentencized.hf"
    elif args.bucketing:
        print(f"{'='*10} Using bucketing batching technique for minimal padding. {'='*10}")
        # for using the padding setting same as in STRLM 2020 GPT-2 expr.
        import warnings
        warnings.warn("This setting constructs batches with sequences with nearly equal lengths for minimal padding. To do so, it will not use the batch size specified in the expr args.")
        dataset_file_name = f"{args.dataset}-no-pad.hf"
    else:
        # the dataset preprocessed with padding for typical batch training.
        dataset_file_name = f"{args.dataset}.hf"
        
    if args.bucketing:
        path_to_bucket = os.path.join(args.dataset_load_dir, f"{args.dataset}-bucketing.pth")
        if os.path.exists(path_to_bucket):
            datasets = torch.load(path_to_bucket)
        else:
            datasets = load_from_disk(os.path.join(args.dataset_load_dir, dataset_file_name))
            datasets = load_bucketing_dataset(datasets, model_config, pad_token_id, args)
            torch.save(datasets, path_to_bucket)
    else:
        datasets = load_from_disk(os.path.join(args.dataset_load_dir, dataset_file_name))
    
    eval_dataloader = DataLoader(
        dataset = datasets[args.eval_split], 
        collate_fn = None if args.bucketing else default_data_collator,
        shuffle = False,
        batch_size = 1 if args.bucketing else args.eval_batch_size,
    )
    
    decode_dataloader = DataLoader(
        dataset = datasets[args.decode_split], 
        collate_fn = None if args.bucketing else default_data_collator,
        shuffle = False if eval_mode else True,
        batch_size = 1 if args.bucketing else args.eval_batch_size,
    )
    
    if eval_mode:
        return eval_dataloader, decode_dataloader
    
    else:
        train_dataloader = DataLoader(
            dataset = datasets["train"], 
            collate_fn = None if args.bucketing else default_data_collator, 
            shuffle = True, 
            batch_size = 1 if args.bucketing else args.train_batch_size,
        )
        return train_dataloader, eval_dataloader, decode_dataloader


def load_bucketing_dataset(dataset_dict, model_config, pad_token_id, args):
    datasets = {}
    for split_name, data in dataset_dict.items():
        datasets[split_name] = BucketingDataset(
            data["input_ids"], split_name, args.context_length, model_config, pad_token_id
        )
    return datasets