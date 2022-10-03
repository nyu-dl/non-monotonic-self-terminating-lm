import argparse
import os
import pprint
import nltk
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2TokenizerFast


def parse_args_and_preprocess(input_args):
    nltk.download('punkt')
    args = parse_args(input_args)
    print(pprint.pformat(vars(args)))
    
    # setup tokenizer.
    if args.use_fast:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    dataset = load_dataset("wikitext", args.dataset)

    # preprocessing method for batch preprocessing. 
    def preprocess_mapper(example):
        # strip the newline ('\n') character.
        texts = map(str.strip, example["text"])
        # remove empty stings.
        texts = filter(lambda x: len(x) > 0, texts)
        
        # need to use BucketingDataset when loading dataset preprocessed with this method.
        if args.no_pad:
            # append <eos> token.
            texts = list(map(lambda x: x + "<|endoftext|>", texts))
            encoded = tokenizer(texts, add_prefix_space=True)
            return {
                "input_ids": encoded["input_ids"],
            }
        else:
            # filter the wikipedia titles/headers w/ "= [title] =" format (de-morgan's law (A and B)^c = A^c or B^c).
            texts = filter(lambda x: x[0] != '=' or x[-1] != '=', texts)
            # sentencize the dataset.
            if args.sentencized:
                texts = [sent for text in list(texts) for sent in nltk.sent_tokenize(text)]
            # halve long sentences.
            elif args.split_long:
                texts = list(texts)
                short_seqs = list(filter(lambda x: len(x.split(" ")) <= 256, texts.copy()))
                long_seqs = list(filter(lambda x: len(x.split(" ")) > 256, texts.copy()))
                long_seq_splits = []
                for seq in long_seqs:
                    sentencized = nltk.sent_tokenize(seq)
                    split_half = len(sentencized) // 2
                    long_seq_splits.append(' '.join(map(str, sentencized[:split_half])))
                    long_seq_splits.append(' '.join(map(str, sentencized[split_half:])))
                long_seq_splits = list(map(str.strip, long_seq_splits))
                texts = long_seq_splits + short_seqs 
        
            # prepend <bos> and append <eos> token.
            texts = list(map(lambda x: x + "<|endoftext|>", texts))
            encoded = tokenizer(texts, add_prefix_space=True, truncation=True, max_length=args.max_length, padding="max_length")

            # Changing the <pad> id (50257) to -100 to ignore <pad> when computing the NLL. 
            labels = torch.LongTensor(encoded["input_ids"])
            labels[labels == 50257] = -100

            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels.tolist(),
            }

    print(f"{'='*80}")
    print(f"{'-' * 10} Preprocessing/tokenizing {args.dataset} {'-' * 10}")
    
    for key in dataset.keys():
        print(f"\n{args.dataset} {key} set size = {len(dataset[key])}")
        dataset[key] = dataset[key].map(
            preprocess_mapper,
            remove_columns=dataset[key].column_names,
            batched=True,
            desc=f"Preprocessing/tokenizing {key} set...",
        )
        print(f"{args.dataset} {key} set size after preprocessing = {len(dataset[key])}")

    print(f"{'-' * 10} Done. {'-' * 10}")
    
    options = ""
    if args.sentencized:
        options += "-sentencized"
    elif args.no_pad:
        options += "-no-pad"

    filename = f"{args.dataset}{options}.hf"
    save_dir = os.path.join(args.save_dir, filename)
    dataset.save_to_disk(save_dir)
    print(f"Dataset saved at: {save_dir}")


def parse_args(input_args:list):
    parser = argparse.ArgumentParser( description="Preprocessing script for wikitext-[103,2] dataset.")
    parser.add_argument(
        "--dataset", type=str, choices=["wikitext-103-raw-v1","wikitext-2-raw-v1"], default="wikitext-103-raw-v1",
    )
    parser.add_argument(
        "--model", type=str, choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="gpt2",
    )
    parser.add_argument("--max-length", type=int, default=368)
    parser.add_argument(
        "--save-dir", type=str, default=os.path.join(os.environ['NMST_DIR'],"data"),
        help="Directory to store the preprocessed dataset.",
    )
    parser.add_argument(
        "--use-fast", type=int, default=0, choices=[0,1], 
        help="Flag for choosing either GPT2Tokenizer (0) or GPT2TokenizerFast (1).",
    )
    parser.add_argument(
        "--sentencized", type=int, default=0, choices=[0,1], 
        help="Flag for spliiting the wikitext dataset by sentence.",
    )
    parser.add_argument(
        "--split-long", type=int, default=0, choices=[0,1], 
        help="Flag for havling seqs. with outlier lengths.",
    )
    parser.add_argument(
        "--no-pad", type=int, default=1, choices=[0,1], 
        help="Flag for minimal preprocessing.",
    )
    args = parser.parse_args()
    return args


def main():
    parse_args_and_preprocess(None)
    
if __name__ == "__main__":
    main()
