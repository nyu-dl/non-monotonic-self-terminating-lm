# A Non-monotonic Self-terminating Language Model

An official repository of the paper [A Non-monotonic Self-terminating Language Model]().

**Authors**: Eugene Choi, Cheolhyoung Lee, Kyunghyun Cho

![Non-monotonic p(eos) plot](/eos.png)

## Requirements

```bash
pip install -r requirements.txt
```

## Environment variable

Set the following environment variable to run training and evaluation scripts seamlessly.

```bash
export NMST_DIR=/path/to/non-monotonic-self-terminating-lm
```

## Running experiments from the paper

All experiments in the paper used a single NVIDIA Quadro RTX 8000.

### WikiText-2
Run `python $NMST_DIR/src/wiki2/train.py` run WikiText-2 expriments from the paper.

### WikiText-103

First, prepare the dataset by running the following commands:
```bash
cd $NMST_DIR/src/gpt2
python preprocess.py
```
Once completed, finetune GPT-2 using `train.py` script.

