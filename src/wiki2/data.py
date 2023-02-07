import torch
import os
import pickle
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple
import utils as utils

# --- data utils 
class Dictionary(object):
    def __init__(
        self,
        dataset,
        include_valid: bool = True,
        special_tokens: list = ["<pad>", "<unk>", "<bos>", "<cpad>", "<eos>"],
    ):
        """This class helps vectorize a text corpus by mapping each word-level token into a unique integer token id.
        
        :param dataset: (wikitext) dataset object from huggingface datasets
        :param include_valid: if True, tokens from the validation set are included 
        :param special_tokens: a list of speical tokens for seq2seq training
        """
        self.tokens = []
        self.ids = {}
        self.counts = {}
        
        for seq in tqdm(dataset["train"]["text"], total=len(dataset["train"])):
            seq = seq.strip()
            if len(seq) != 0: # discard empty seq.
                for w in seq.split(" "):
                    self.add_token(w)
        
        if include_valid:
            for seq in tqdm(dataset["validation"]["text"], total=len(dataset["validation"])):
                seq = seq.strip()
                if len(seq) != 0: # discard empty seq.
                    for w in seq.split(" "):
                        self.add_token(w)
        
        # Add special tokens (at the end so we can optionally exclude them in the output space)
        # Always put eos last, since it helps the RNNLanguageModelST w/ indexing during foward step.
        assert len(special_tokens) > 0 and special_tokens[-1] == "<eos>"
        
        for token in special_tokens:
            self.add_token(token)

    def add_token(self, word: str):
        if word not in self.tokens:
            self.tokens.append(word)
            self.counts[word] = 0
            _w_id = len(self.tokens) - 1
            self.ids[word] = _w_id
        else:
            self.counts[word] += 1 # self.counts: a dict keeping track of a word occurence frequency.

    def get_id(self, word: str) -> int:
        return self.ids[word] # self.ids: a dict mapping a word to its token_id.

    def get_token(self, idx: int) -> str:
        return self.tokens[idx] # self.tokens: a list mapping a token_id to correponding word.

    def decode_idx_seq(self, ids: list) -> list:
        return [self.tokens[idx] for idx in ids]

    def encode_token_seq(self, seq: list) -> list:
        return [self.ids[word] if word in self.ids else self.ids["<unk>"] for word in seq] # maps tokens in a seq. to token_ids if they exists in vocab.
    
    def __len__(self):
        return len(self.tokens)


class LMDataset(Dataset):
    def __init__(
        self, 
        list_of_token_lists: list,
    ):
        """This class processes vectorized text corpus into (input,target) pairs for LM training and stores them as list of tensors.
        
        :param list_of_token_lists: a list of vectorized sequences
        """
        self.input_tensors = []
        self.target_tensors = []

        """
        For seq2seq training, given a sequence, [<bos>, w_1, ..., w_T, <eos>], we simply process it like:
        input = [<bos>, w_1, ..., w_T]
        label = [w_1, ..., w_T, <eos>].
        """
        for sample in list_of_token_lists:
            self.input_tensors.append(torch.LongTensor([sample[:-1]]))
            self.target_tensors.append(torch.LongTensor([sample[1:]]))

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx: int):
        return self.input_tensors[idx], self.target_tensors[idx]


def get_dataloaders_and_vocab(args) -> Tuple[dict, Dictionary]:
    """Preprocesses/loads preprocessed dataset and returns a dict with data splits as keys and correponding dataloaders as values.
    
    :param args: argsparse.Namespace obj containing preprocessing options
    :returns: a dict of split names as keys and correponding dataloaders as values, a dictionary mapping
    """
    datasets, vocab = load_tokenized_dataset(args)

    # Add random contexts for evaluation. (given a random prefix, does our model terminate?)
    datasets["random"] = random_prefixes(
        vocab, args.mask_context_k, args.num_random_prefixes,
    )
    
    pad_idx = vocab.get_id("<pad>")
    
    data_loaders = {
        name: DataLoader(
            datasets[name],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: pad_collate_fn(x, pad_idx),
        )
        for name in datasets
    }
    
    return data_loaders, vocab


def load_tokenized_dataset(args) -> Tuple[dict, Dictionary]:
    """Preprocesses dataset into word-level tokens, or loads a dataset if a processed version exists.
    
    :param args: argparse.Namespace object containing expr args.
    :returns: a dict containing preprocessed dataset at word-level, a dictionary mapping of tokens to their unique ids.
    """
    vocab_file = f"{args.dataset_version}-vocab.pkl"
    vocab_path = os.path.join(args.dataset_load_dir, vocab_file)
    
    if os.path.exists(vocab_path):
        vocab = pickle.load(open(vocab_path, 'rb'))
    else:
        raw_datasets = load_dataset("wikitext", args.dataset_version)
        vocab = Dictionary(raw_datasets, include_valid=True)
        os.makedirs(args.dataset_load_dir, exist_ok=True)
        pickle.dump(vocab, open(vocab_path, 'wb'))

    dataset_file = f"{args.dataset_version}-dataset.pth"
    dataset_path = os.path.join(args.dataset_load_dir, dataset_file)

    if os.path.exists(dataset_path):
        datasets = torch.load(dataset_path)
    else:
        raw_datasets = load_dataset("wikitext", args.dataset_version)
        tokenized_datasets = word_tokenize(raw_datasets, vocab, min_context_length=args.mask_context_k)
        datasets = {name: LMDataset(ds) for name, ds in tokenized_datasets.items()}
        torch.save(datasets, dataset_path)

    return datasets, vocab


def random_prefixes(
    vocab: Dictionary,
    context_len: int,
    num_prefixes: int,
) -> LMDataset:
    """Generates random contexts to test if our model terminates when prompted with o.o.d. context samples.
    
    :param vocab: a Dictionary object with a list of valid token_ids.
    :param context_len: the length of each context.
    :param num_prefixes: the number of contexts to generate.
    :returns: a LMDataset object containing randomly generated contexts.
    """
    if num_prefixes == -1:
        num_prefixes = 1000
    prefixes = torch.randint(len(vocab), (num_prefixes, context_len))
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<cpad>"]
    for tok in special_tokens:
        if tok in vocab.counts:
            id_ = vocab.get_id(tok)
            prefixes[prefixes == id_] = 6

    bos = torch.tensor([vocab.get_id("<bos>")] * num_prefixes).unsqueeze(1)
    prefixes = torch.cat((bos, prefixes), 1).tolist()
    dataset = LMDataset(prefixes)
    return dataset


def word_tokenize(
    datasets, 
    dictionary: Dictionary,
    min_context_length: int = 0, 
    discard_seq_less_than_context_length: bool = False,
) -> dict:
    """A method for tokenizing a given dataset into word-level tokens.
    
    :param dataset: (wikitext) dataset object from huggingface datasets
    :param dictionary: a dictionary containing mappings of word-level tokens to their token ids.
    :param min_context_length: length of a context used as a prompt in seq. completion task.
    :param discard_seq_less_than_context_length: if True, the method discards all sequences shorter than `min_context_length`.
    :return: A dictionary of tokenzied datasets.
    """
    tokenized_datasets = {}
    for split in datasets.keys():
        _current_dictified = []
        for seq in tqdm(datasets[split]["text"], total=len(datasets[split]), desc=f"{split:6s}"):
            seq = seq.strip().split(" ")
            if len(seq) < min_context_length and not discard_seq_less_than_context_length:
                num_pad = min_context_length - len(seq)
                seq = ["<cpad>"] * num_pad + seq
            seq = ["<bos>"] + seq + ["<eos>"]         
            encoded_l = dictionary.encode_token_seq(seq)
            _current_dictified.append(encoded_l)
        tokenized_datasets[split] = _current_dictified
    return tokenized_datasets


def pad_collate_fn(
    batch: list,
    pad_idx: int, 
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """A data collator method that forms a batch by padding a list of vectorized sequences to equal lengths.
    
    :param batch: A list of input/target pairs of vectorized sequences.
    :param pad_token: Padding token index.
    :return: A input/target pairs of vectorized sequences in torch.LongTensor form of size [batch_size, max_seq_len].    
    """
    input_list = [s[0] for s in batch]
    target_list = [s[1] for s in batch]
    input_tensor = pad_list_of_tensors(input_list, pad_idx)
    target_tensor = pad_list_of_tensors(target_list, pad_idx)
    return input_tensor, target_tensor


def pad_list_of_tensors(
    list_of_tensors: list,
    pad_token: int,
) -> torch.LongTensor:
    """A helper function for `pad_collate_fn`. Supports dynamic padding (NO truncation).

    :param list_of_tensors: A list of vectorized sequences. Each sequence is in torch.LongTensor format.
    :param pad_token: Padding token index.
    :return: A torch.LongTensor of size [batch_size, max_seq_len].
    """
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    for t in list_of_tensors:
        padded_tensor = torch.cat([t,torch.LongTensor([[pad_token]*(max_length - t.size(-1))]),], dim=-1)
        padded_list.append(padded_tensor)
    padded_tensor = torch.cat(padded_list, dim=0)
    return padded_tensor

