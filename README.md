# A Non-monotonic Self-terminating Language Model

The official repository of the ICLR 2023 conference paper, **"A Non-monotonic Self-terminating Language Model"**.

**Authors**: [Eugene Choi](https://eugene-choi.github.io/), [Kyunghyun Cho](https://www.kyunghyuncho.me/), [Cheolhyoung Lee](https://sites.google.com/view/cheolhyounglee)

[[ArXiv](https://arxiv.org/abs/2210.00660)] [[Openreview](https://openreview.net/forum?id=vw-5EgYbJZr)]


<!--- ![Non-monotonic p(eos) plot](/eos.png) --->
<img src="https://github.com/nyu-dl/non-monotonic-self-terminating-lm/blob/main/eos.png" width=70% height=70%>

## 1. Overview:
The repository is organized as follows.
```!
.
├── eos.png
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── gpt2
    │   ├── data.py
    │   ├── eval.py
    │   ├── metrics.py
    │   ├── nmst.py
    │   ├── preprocess.py
    │   ├── st.py
    │   ├── train.py
    │   └── utils.py
    └── wiki2
        ├── data.py
        ├── decoding_utils.py
        ├── evaluate.py
        ├── model_utils.py
        ├── train.py
        └── utils.py
```

## 2. Setup:
The code was written in `Python 3.9.12` and all experiments in the paper were conducted using a single `NVIDIA Quadro RTX 8000`. Please set the environment variable and install dependencies, following commands below. This will ensure a seamless execution of the training and evaluation scripts:

### Dependencies:
```bash!
pip install -r requirements.txt
```
### Environment variable:
```bash!
export NMST_DIR=/path/to/non-monotonic-self-terminating-lm
```

## 3. Running experiments:

### WikiText-2

--- 

Make sure that you are in the following directory.
```bash!
cd ${NMST_DIR}/src/wiki2
```
### Training:

Note: The default `argparse` configuration for `train.py` in `${NMST_DIR}/src/wiki2` is based on the hyperparameters used in the `NMST+LSTM (1e-4)` experiment. The epsilon value for `NMST+` and `ST+` can be set to any value in (0,1).

Please refer to the following example commands comparing `NMST/ST/VA` parameterizations with an epsilon value of `1e-5` as a guide for running the `WikiText-2` experiments.

**LSTM**:
- `VA+LSTM`
```!
python train.py --loss mle --rnn-type nn.LSTM --dropout 0.5 --embedding-dim 512 --num-layers 2 --hidden-size 512 --rnn-dropout 0.0 --batch-size 32 --expr-name lstm_lm
```
- `ST+LSTM (1e-5)`
```!
python train.py --loss st --epsilon 1e-5 --rnn-type nn.LSTM --dropout 0.5 --embedding-dim 512 --num-layers 2 --hidden-size 512 --rnn-dropout 0.0 --batch-size 32 --expr-name lstm_st-1e-5
```
- `NMST+LSTM (1e-5)`
```!
python train.py --loss nmst --epsilon 1e-5 --rnn-type nn.LSTM --dropout 0.5 --embedding-dim 512 --num-layers 2 --hidden-size 512 --rnn-dropout 0.0 --batch-size 32 --expr-name lstm_nmst-1e-5
```

#### RNN:
- `VA+RNN`
```!
python train.py --loss mle --rnn-type nn.RNN --dropout 0.3 --embedding-dim 256 --num-layers 2 --hidden-size 256 --rnn-dropout 0.0 --batch-size 32 --expr-name rnn_lm
```
- `ST+RNN (1e-5)`
```!
python train.py --loss st --epsilon 1e-5 --rnn-type nn.RNN --dropout 0.3 --embedding-dim 256 --num-layers 2 --hidden-size 256 --rnn-dropout 0.0 --batch-size 32 --expr-name rnn_st-1e-5
```
- `NMST+RNN (1e-5)`
```!
python train.py --loss nmst --epsilon 1e-5 --rnn-type nn.RNN --dropout 0.3 --embedding-dim 256 --num-layers 2 --hidden-size 256 --rnn-dropout 0.0 --batch-size 32 --expr-name rnn_nmst-1e-5
```

### Inference:
```!
python evaluate.py --model-load-dir ${NMST_DIR}/checkpoint/wiki2/MODEL_DIR --DECODING_METHOD 1
```
The `evaluate.py` scipt supports multiple decoding methods, including greedy decoding, ancestral sampling, top-k sampling, nucleus sampling, beam search, as well as consistent top-k and consistent nucleus sampling, as proposed in the paper ["Consistency of a Recurrent Language Model with Respect to Incomplete Decoding."](https://aclanthology.org/2020.emnlp-main.448/) Please make sure to choose the appropriate decoding hyperparameters (such as k in top-k sampling or beam size for beam search) before running the inference.


### WikiText-103

---

First, prepare the dataset by running the following commands:
```bash!
cd ${NMST_DIR}/src/gpt2
python preprocess.py
```

### Training:
Once completed, finetune `GPT-2 (124M)` using `train.py` script:
- `VA+GPT2`
```!
python train.py --loss mle --expr-name lm --model-name gpt2 --bucketing 1 --eval 1 --decode 1
```
- `ST+GPT2 (1e-5)`
```!
python train.py --loss st --epsilon 1e-5 --expr-name st_1e-5 --model-name gpt2 --bucketing 1 --eval 1 --decode 1
```
- `NMST+GPT2 (1e-5)`
```!
python train.py --loss nmst --epsilon 1e-5 --expr-name nmst_1e-5 --model-name gpt2 --bucketing 1 --eval 1 --decode 1
```

### Inference:
```!
python eval.py --model-load-dir ${NMST_DIR}/checkpoint/gpt2/MODEL_DIR --DECODING_METHOD 1 --expr-name EVAL_RUN_NAME
```
The `eval.py` scipt supports greedy decoding, ancestral sampling, top-k sampling, nucleus sampling, and beam search. Please make sure to choose the appropriate decoding hyperparameters (such as k in top-k sampling or beam size for beam search) before running the inference.


## 4. Logs:
The Weights and Biases logs for all experimental results can be accessed through the links provided below:

### WikiText-2:

#### Training:
https://wandb.ai/eugenechoi/nmst-rnn/workspace?workspace=user-eugenechoi

(Note: You can filter the logs to view only the `RNN` or `LSTM` experiments by selecting `rnn_type` from the filter dropdown menu.)

### WikiText-103:
#### Training:
https://wandb.ai/eugenechoi/nmst-gpt2/workspace?workspace=user-eugenechoi

#### Decoding:
- **Perplexity & Greedy**:
https://wandb.ai/eugenechoi/nmst-gpt2-greedy/workspace?workspace=user-eugenechoi

- **Nucleus**:
https://wandb.ai/eugenechoi/nmst-gpt2-topp/workspace?workspace=user-eugenechoi

- **Top-k**:
https://wandb.ai/eugenechoi/nmst-gpt2-topk/workspace?workspace=user-eugenechoi

- **Beam Search**:
https://wandb.ai/eugenechoi/nmst-gpt2-beam/workspace?workspace=user-eugenechoi


## BibTex:

Please use the following bib to cite our work:
```!
@inproceedings{
choi2023a,
title={A Non-monotonic Self-terminating Language Model},
author={Eugene Choi and Kyunghyun Cho and Cheolhyoung Lee},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=vw-5EgYbJZr}
}
```

