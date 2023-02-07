import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- model utils
class VanillaLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.eos_idx = args.eos_idx
        self.drop = nn.Dropout(args.dropout)
        self.lookup = nn.Embedding(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            padding_idx=args.pad_idx,
        )
        if args.rnn_type == "nn.LSTM":
            self.rnn = eval(args.rnn_type)(
                input_size = args.embedding_dim,
                hidden_size = args.hidden_size,
                proj_size = 0 if args.hidden_size == args.embedding_dim else args.embedding_dim,
                num_layers = args.num_layers,
                dropout = args.rnn_dropout,
                batch_first = True,
            )
        else:
            self.rnn = eval(args.rnn_type)(
                input_size = args.embedding_dim,
                hidden_size = args.hidden_size,
                num_layers = args.num_layers,
                dropout = args.rnn_dropout,
                batch_first = True,
            )
        self.projection = nn.Linear(args.embedding_dim, args.output_dim, bias=False)
        if args.tie_weights:
            self.projection.weight = self.lookup.weight

    def forward(self, encoded_input_sequence, hidden=None, return_all=False):
        logits = self.step(encoded_input_sequence, hidden, return_all=return_all)
        return logits

    def step(self, encoded_input_sequence, hidden, return_all=False):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        h_all, h_last = self.rnn(embeddings, hidden)
        logits = self.projection(self.drop(h_all))
        if return_all:
            return logits, h_last
        else:
            return logits
    
    
class SelfTerminatingLM(VanillaLM):
    def __init__(self, args):
        super().__init__(args)
        self.epsilon = args.epsilon
        self.context_length = args.mask_context_k
        self.smoothing_eps = 1e-20
    
    # for training
    def forward(
        self, 
        encoded_input_sequence, 
        return_all=False, 
        includ_zero=False,
    ):
        """
        :param includ_zero: exact replication of Welleck et. al. 2020 implementation which included hidden state for t=0
        """
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        h_all, h_last = self.rnn(embeddings)
        logits = self.projection(self.drop(h_all))
        
        v_logits = logits[...,:self.eos_idx]
        eos_logits = logits[...,self.eos_idx].unsqueeze(-1)
        
        bound = 1 - self.epsilon
        betas = torch.clamp(bound * eos_logits.sigmoid(), min=self.smoothing_eps)
        
        # Enforce p(eos) = 0 throughout the context.
        if self.context_length > 0:
            betas[:, :self.context_length] = 1
            
        """Numerically stable p(eos):   p = 1 - prod beta
                              => log(1-p) = sum log beta
                              =>        p = 1 - exp sum log beta
        """
        if includ_zero:
            p_eoss = 1.0 - betas.log().cumsum(dim=1).exp()
            p_eos_prev = torch.zeros_like(p_eoss[:,:1,:1])
            p_eos_prevs = torch.cat((p_eos_prev, p_eoss), 1)[:, :-1, :]
            alphas = torch.clamp(betas * (1-p_eos_prevs), min=self.smoothing_eps)
        else:
            alphas = betas.log().cumsum(dim=1).exp()        
            p_eoss = 1.0 - alphas

        p_Vs = alphas * v_logits.softmax(dim=-1)
        ps = torch.cat((p_Vs, p_eoss), dim=2) 
        
        ps = torch.clamp(ps, min=self.smoothing_eps) # smoothe the output distribution.
        ps = ps / ps.sum(-1, keepdim=True) # re-normalize the distribution after clamping/smoothing
        log_ps = ps.log()
        if return_all:
            return log_ps, h_last, p_eoss[:, -1:, :]
        else:
            return log_ps
    
    # for evaluation
    def step(
        self, 
        encoded_input_sequence, 
        hidden, 
        p_eos_prev=None, 
        return_all=False,
    ):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        h_all, h_last = self.rnn(embeddings, hidden)
        logits = self.projection(self.drop(h_all))
            
        v_logits = logits[...,:self.eos_idx]
        eos_logits = logits[...,self.eos_idx].unsqueeze(-1)
        
        bound = 1 - self.epsilon # (1-ε)
        betas = torch.clamp(bound * eos_logits.sigmoid(), min=self.smoothing_eps)
        
        # -- Difference for the step case:
        alphas = betas * (1.0 - p_eos_prev)
        
        p_eos = 1.0 - alphas
        p_Vs = alphas * v_logits.softmax(-1)
        
        ps = torch.cat((p_Vs, p_eos), dim=2)
        ps = torch.clamp(ps, min=self.smoothing_eps)
        ps = ps / ps.sum(-1, keepdim=True)
        log_ps = ps.log()
        
        return log_ps, h_last, p_eos
        
        
class NonMonotonicSTLM(SelfTerminatingLM):
    def __init__(self, args):
        super().__init__(args)
        self.log_bounds = math.log(1.0 - args.epsilon)
        self.smoothing_eps = 1e-35
        
    def forward(self, encoded_input_sequence, return_all=False):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        h_all, h_last = self.rnn(embeddings)
        logits = self.projection(self.drop(h_all))
        
        v_lprobs = logits[...,:self.eos_idx].log_softmax(-1)
        eos_logits = logits[...,self.eos_idx].unsqueeze(-1)
        
        # Enforce p(eos) = 0 throughout the context.
        if self.context_length > 0:
            context_log_lb = torch.ones_like(eos_logits[:,:self.context_length])
            log_lb = torch.full_like(eos_logits[:,self.context_length:], self.log_bounds).cumsum(dim=1)
            log_lb = torch.cat([context_log_lb, log_lb], dim=1)
            f_lower_bound = 1.0 - log_lb.exp()
            f_lower_bound[:,:self.context_length] = self.smoothing_eps
        else:
            log_lb = torch.full_like(eos_logits, self.log_bounds).cumsum(dim=1)
            f_lower_bound = 1.0 - log_lb.exp()


        # log(σ(z^{<eos>}_t))
        log_eos_ub_score = F.logsigmoid(eos_logits)
        # log(1-σ(z^{<eos>}_t))
        log_eos_lb_score = F.logsigmoid(-eos_logits)

        """v_lprobs
        1 - p = 1 - [(1-(1-ε)^t)(1 - σ) + σ]
              = 1 - (1-(1-ε)^t) + (1-(1-ε)^t)σ - σ = (1-ε)^t + σ - (1-ε)^t)σ - σ
              = (1-ε)^t - σ*(1-ε)^t
              = (1 - σ)(1-ε)^t
        => log(1 - p) = log(1 - σ) + t*log(1-ε)
        """
        v_lprobs = log_lb + log_eos_lb_score + v_lprobs
        
        # log(1-σ(z^{<eos>}_t)*f_low(t))
        log_lb_eos = f_lower_bound.log() + log_eos_lb_score 
        
        """
        addition in log-space:
        log(α+β) = log(α) - log(σ(log(α) - log(β)))
        => log(p(eos)) = log(f_lb) - log(σ(f_lb - f_ub))
        """
        lprob_eos = log_lb_eos - F.logsigmoid(log_lb_eos - log_eos_ub_score)
        log_ps = torch.cat((v_lprobs, lprob_eos), dim=-1)
        log_ps = log_ps - log_ps.logsumexp(-1, keepdim=True)
        if return_all:
            return log_ps, h_last, log_lb
        else:
            return log_ps
    
    # for evaluation
    def step(
        self, 
        encoded_input_sequence, 
        hidden, 
        p_eos_prev=None, 
        return_all=False,
    ):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        h_all, h_last = self.rnn(embeddings, hidden)
        logits = self.projection(self.drop(h_all))
        
        v_lprobs = logits[...,:self.eos_idx].log_softmax(-1)
        eos_logits = logits[...,self.eos_idx].unsqueeze(-1)
        
        # -- Difference for the step case:
        log_lb = torch.full_like(eos_logits, self.log_bounds) + p_eos_prev
            
        f_lower_bound = 1.0 - log_lb.exp()
        # log(σ(z^{<eos>}_t))
        log_eos_ub_score = F.logsigmoid(eos_logits)
        # log(1-σ(z^{<eos>}_t))
        log_eos_lb_score = F.logsigmoid(-eos_logits)
        
        
        """v_lprobs
        1 - p = 1 - [(1-(1-ε)^t)(1 - σ) + σ]
              = 1 - (1-(1-ε)^t) + (1-(1-ε)^t)σ - σ = (1-ε)^t + σ - (1-ε)^t)σ - σ
              = (1-ε)^t - σ*(1-ε)^t
              = (1 - σ)(1-ε)^t
        => log(1 - p) = log(1 - σ) + t*log(1-ε)
        """
        v_lprobs = log_lb + log_eos_lb_score + v_lprobs
        
        # log(1-σ(z^{<eos>}_t))+log(f_lb(t))
        log_lb_eos = f_lower_bound.log() + log_eos_lb_score
        """
        addition in log-space:
        log(α+β) = log(α) - log(σ(log(α) - log(β)))
        => log(p(eos)) = log(f_lb) - log(σ(f_lb - f_ub))
        """
        lprob_eos = log_lb_eos - F.logsigmoid(log_lb_eos - log_eos_ub_score)
        log_ps = torch.cat((v_lprobs, lprob_eos), dim=-1)
        log_ps = log_ps - log_ps.logsumexp(-1, keepdim=True)
        return log_ps, h_last, log_lb
