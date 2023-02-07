import argparse
import pickle
import os
import tqdm
import torch
import numpy as np
import model_utils
import data
from typing import Union

def decoding_dataset_stats(
    model: torch.nn.Module,
    dataloaders: dict,
    vocab: data.Dictionary, 
    device: torch.device,
    num_samples: dict = {},
    max_steps: int = 500,
    temperature: float = 1.0,
    prefix_length:int = 5,
    decoding: tuple = ("greedy", "sample"),
    consistent_sampling=False,
    save_decoding_logs=False,
):
    """A method for decoding continuations from models given prefixes, and computing their non-termination stats.
    
    """
    results = {}

    def _stats(sequences):
        # `uniq_nonterminated` is the ratio of unique nonterminated decodings
        non_terminated = [tuple(x) for x in sequences if len(x) == max_steps]
        s = {
            "nonterminated": len(non_terminated) / len(sequences),
            "uniq_nonterminated": len(set(non_terminated)) / max(len(non_terminated), 1),
            "avg_len": np.mean([len(x) for x in sequences]),
        }
        return s

    def _to_text(prefixes, res, targets, vocab):
        pr_texts = []
        cont_texts = []
        target_texts = []
        for pr, cont, target in zip(prefixes, res, targets):
            pr_text = vocab.decode_idx_seq(pr)
            cont_text = vocab.decode_idx_seq(cont)
            target_text = vocab.decode_idx_seq(target)
            pr_texts.append(pr_text)
            cont_texts.append(cont_text)
            target_texts.append(target_text)
        
        return (pr_texts, cont_texts, target_texts) 

    print("Computing decoding stats...")
    for name, data_loader in dataloaders.items():
        results[name] = {}
        for decoding_algo in decoding:
            print(f"dataset {name}\tdecoding {decoding_algo}")
            res, prefixes, targets = decode_dataset(
                model,
                data_loader,
                vocab.get_id("<bos>"),
                vocab.get_id("<eos>"),
                num_samples.get(name, -1),
                max_steps,
                decoding_algo=decoding_algo,
                device=device,
                prefix_length=prefix_length,
                temperature=temperature,
                consistent_sampling=consistent_sampling,
            )
            
            key = decoding_algo if isinstance(decoding_algo, str) else f"{decoding_algo[0]}_{str(decoding_algo[1])}"
            results[name][key] = _stats(res)
            if save_decoding_logs:
                results[name][key]['decoding_logs'] = _to_text(prefixes, res, targets, vocab)
    return results


def compute_ppl_dataloader(
    args: argparse.Namespace,
    vocab: data.Dictionary, 
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    dataset_split: str,
    device: torch.device,
) -> dict:
    """A method for computing perplexity given a model and a dataset.
    
    """
    eos_idx = vocab.get_id("<eos>")
    pad_idx = vocab.get_id("<pad>")
    sum_loss =  0.0   
    sum_num_pred_tokens = 0
    
    iterator = tqdm.auto.tqdm(enumerate(data_loader), total = len(data_loader))
    model.eval()
    
    with torch.no_grad():
        for i, (inp, target) in iterator:
            inp = inp.to(device)
            target = target.to(device)
            output = model(inp)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                
            # optionally we mask loss computed over the context C here
            if args.mask_context_k > 0:
                loss = loss.view(target.size())
                loss[:, : args.mask_context_k] = 0.0

            num_context_tokens = target.size(0) * args.mask_context_k
            num_pred_tokens = target.ne(pad_idx).count_nonzero().item() - num_context_tokens
            sum_num_pred_tokens += num_pred_tokens
            
            sum_loss += loss.sum()
 
        avg_loss = sum_loss / sum_num_pred_tokens
        log_dict={
            f"{dataset_split}/avg_loss": avg_loss.item(),
            f"{dataset_split}/ppl": avg_loss.exp().item(),
        }
    return log_dict

def decode_dataset(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    bos_token: int,
    eos_token: int,
    num_samples: int,
    max_steps: int,
    decoding_algo: Union[str, tuple],
    device: torch.device,
    prefix_length: int = 0,
    consistent_sampling:bool = False,
    temperature:float = 1.0,
):
    with torch.no_grad():
        xs = []
        prefixes = []
        targets = []
        
        iterator = tqdm.auto.tqdm(
            enumerate(data_loader), 
            total = len(data_loader) if num_samples == -1 else num_samples // data_loader.batch_size
        )
        
        for minibatch_id, (inp, target) in iterator:
            inp = inp.to(device)

            hidden = None
            p_eos_prev = None
            eos_hidden = None
            
            # -- encode the prefix (context) tokens
            prefix = inp[:, : prefix_length + 1]  # +1 for <bos>
            if isinstance(model, model_utils.SelfTerminatingLM):
                output, hidden, p_eos_prev = model(prefix, return_all=True)
            else:
                output, hidden = model.step(prefix, hidden, return_all=True)
            
            # -- generate!
            if 'beam' in decoding_algo:
                beam_size = int(decoding_algo.split("_")[1])  # e.g. beam_4
                batch_size = inp.size(0)
                max_timestep = max_steps

                count_finished = torch.zeros_like(batch_size)
                
                finished_hypotheses = {
                    i:[] for i in range(batch_size)
                }
                finished_scores = {
                    i:[] for i in range(batch_size)
                }

                # first beam iteration is out of the loop here
                if isinstance(model, model_utils.SelfTerminatingLM):
                    log_probs = output[:, -1, :]
                else:
                    log_probs = output[:, -1, :].log_softmax(dim=-1)  # [batch_size, vocab_size]
                    
                vocab_size = log_probs.size(-1)

                top_scores, top_tokens = torch.topk(log_probs, beam_size, dim=-1, largest=True, sorted=True)
                
                # we add to finished even now when eos is selected too
                current_eos_mask = (top_tokens == eos_token)
                count_finished = count_finished + current_eos_mask.sum(1).long()

                # we need this loop... ?
                for beam_id, beam_eos_mask in enumerate(current_eos_mask):
                    if any(beam_eos_mask):
                        finished_in_this_beam = top_tokens[beam_id, beam_eos_mask]
                        finished_scores_in_this_beam = top_scores[beam_id, beam_eos_mask]
                        finished_hypotheses[beam_id].extend([finished_in_this_beam.tolist()])
                        finished_scores[beam_id].extend(finished_scores_in_this_beam.tolist())
                
                hypotheses = [
                    (
                        top_tokens[:,:,None],
                        top_scores,
                        torch.zeros_like(top_tokens)
                    )
                ]

                # expanding the hidden tuple up to the beam_size
                if isinstance(hidden, tuple):  # LSTM
                    expanded_hidden = [None,None]
                    for i in range(2):
                        expanded_hidden[i] = hidden[i][:,:,None,:].expand(-1,-1,beam_size,-1).reshape(2,batch_size*beam_size,-1).contiguous()
                    expanded_hidden = tuple(expanded_hidden)
                else:
                    expanded_hidden = hidden[:,:,None,:].expand(-1,-1,beam_size,-1).reshape(2,batch_size*beam_size,-1).contiguous()

                # input for the first beam timestep
                expanded_input = top_tokens.view(batch_size*beam_size,1)
                
                if isinstance(model, model_utils.SelfTerminatingLM):
                    p_eos_prev = p_eos_prev[:,-1:,:].repeat(beam_size, 1,1)
                for timestep in range(1, max_timestep):
                    # change below should be enough for the STRNN
                    # initial_states = [state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous() for state in hidden_state]
                    if isinstance(model, model_utils.RNNLanguageModelST):
                        expanded_output, expanded_hidden, p_eos_prev = model.step(expanded_input, expanded_hidden, p_eos_prev[:,-1:,:])
                    else:
                        expanded_output, expanded_hidden = model.step(expanded_input, expanded_hidden)
                    # torch.cuda.empty_cache()
                    # reshaping back as batch_size * beam_size
                    decoupled_output = expanded_output[:, None, :,:].view(batch_size, beam_size, 1, -1)  # (batch, beam, 1, vocab)
                    # del expanded_output
                    # -> log_softmax
                    decoupled_output = torch.log_softmax(decoupled_output, dim=-1)

                    partial_from_prev_timestep = hypotheses[timestep-1][0]  # index 0 is partial
                    scores_from_prev_timestep = hypotheses[timestep-1][1]  # index 1 is scores
                    # partial_from_prev_timestep

                    # check for eos, do not select anything after eos
                    eos_mask = partial_from_prev_timestep[:,:,-1] == eos_token
                    scores_from_prev_timestep[eos_mask] = -10e15
                    
                    # decoupled_output = decoupled_output.to(device)
                    scores_from_prev_timestep = scores_from_prev_timestep.to(device)
                    
                    extended_scores = decoupled_output.add(scores_from_prev_timestep[:,:,None,None])
                    # del decoupled_output
                    # del scores_from_prev_timestep
                    # del eos_mask
                    # coupling it beam*vocab for topk
                    coupled_extended_scores = extended_scores.view(batch_size, beam_size*vocab_size)
                    top_scores, top_ids = torch.topk(coupled_extended_scores, beam_size, dim=-1, largest=True, sorted=True)
                    
                    # del coupled_extended_scores
                    # del extended_scores
                    actual_word_ids = top_ids % vocab_size
                    
                    # make a new input for next iteration

                    expanded_input = actual_word_ids.view(batch_size*beam_size, -1)

                    # prev_hyp_id_per_sample = top_ids // vocab_size
                    prev_hyp_id_per_sample = torch.div(top_ids, vocab_size, rounding_mode='floor')
                    
                    # del top_ids
                    prev_hyp_id_flat = ((torch.arange(batch_size, device=device) * beam_size)[:,None] + prev_hyp_id_per_sample).view(-1)
                    # print(partial_from_prev_timestep)
                    partial_from_prev_timestep = partial_from_prev_timestep.to(device)
                    reordered_prev_hypotheses = torch.index_select(partial_from_prev_timestep.view(batch_size*beam_size,-1), dim=0, index=prev_hyp_id_flat).view(batch_size, beam_size, -1)
                    extended_current_hypotheses = torch.cat([reordered_prev_hypotheses, actual_word_ids[:,:,None]], dim=2)
                    
                    # del partial_from_prev_timestep
                    # del reordered_prev_hypotheses
                    # check currently extended hyps for eos
                    current_eos_mask = (actual_word_ids == eos_token)
                    count_finished = count_finished + current_eos_mask.sum(1).long()
                    # del actual_word_ids
                    # we need this loop... ?
                    for beam_id, beam_eos_mask in enumerate(current_eos_mask):
                        if any(beam_eos_mask):
                            finished_in_this_beam = extended_current_hypotheses[beam_id, beam_eos_mask, :]
                            finished_scores_in_this_beam = top_scores[beam_id, beam_eos_mask]
                            finished_hypotheses[beam_id].extend(finished_in_this_beam.tolist())
                            finished_scores[beam_id].extend(finished_scores_in_this_beam.tolist())
                    # del current_eos_mask
                    # reorder the hidden state
                    if isinstance(expanded_hidden, tuple):  # LSTM
                        new_expanded_hidden = [None,None]
                        num_layers = expanded_hidden[0].size(0)
                        for i in range(num_layers):
                            
                            new_expanded_hidden[i] = torch.index_select(expanded_hidden[i], dim=1, index=prev_hyp_id_flat)
                        new_expanded_hidden = tuple(expanded_hidden)
                    else:
                        new_expanded_hidden = torch.index_select(expanded_hidden, dim=1, index=prev_hyp_id_flat)
                    expanded_hidden = new_expanded_hidden
                    # del new_expanded_hidden
                    
                    # add new hypotheses to beam
                    hypotheses.append(
                        (extended_current_hypotheses.cpu(), top_scores.cpu(), prev_hyp_id_per_sample.cpu())
                    )
                    
                    # del top_scores
                    # del prev_hyp_id_per_sample
                    # del prev_hyp_id_flat
                    # del extended_current_hypotheses
                    # torch.cuda.empty_cache()
                    
                    # check if we have enough ( at least 1) finished for each sample in mini batch
                    # ideally one would do at least beam size, with 1 avg len might be shorter
                    if all(count_finished > 0):
                        break
                
                # now we check what hypotheses are finished
                best_finished_seqs = []
                for beam_id in range(batch_size):
                    if count_finished[beam_id].item() == 0:
                        # non-terminated here
                        # take the first seq from the beam
                        seq = hypotheses[-1][0][beam_id][0].cpu().tolist()
                    else:
                        # find the best one w.r.t score
                        finished_here = finished_scores[beam_id]
                        best_finished_id = np.array(finished_here).argmax()
                        seq = finished_hypotheses[beam_id][best_finished_id]
                    best_finished_seqs.append(seq) 
                x = best_finished_seqs

            else:
                # -- decode
                x = []
                p_eoss = []
                output = output[:,-1:,:]
                eps = None
                
                for t in range(max_steps):
                    if decoding_algo == "greedy":
                        xt = output.argmax(-1)
                    elif decoding_algo == "sample":
                        if isinstance(model, model_utils.SelfTerminatingLM) and temperature != 1.0:
                            raise NotImplementedError("SelfTerminatingLM and NonMonotonicSTLM only supports temperature = 1.0")
                        elif isinstance(model, model_utils.SelfTerminatingLM):
                            xt = output.exp().squeeze(1).multinomial(1)
                        else:
                            xt = (output / temperature).softmax(-1).squeeze(1).multinomial(1)
                    elif isinstance(decoding_algo, tuple):
                        if decoding_algo[0] == "topk":
                            output = top_k_top_p_filtering(
                                output.squeeze(1), 
                                top_k = decoding_algo[1], 
                                consistent_sampling = consistent_sampling, 
                                eos_idx = model.eos_idx
                            )
                        elif decoding_algo[0] == "topp":
                            output = top_k_top_p_filtering(
                                output.squeeze(1), 
                                top_p = decoding_algo[1], 
                                consistent_sampling = consistent_sampling,
                                eos_idx = model.eos_idx
                            )
                        xt = output.softmax(-1).multinomial(1) 
                        # topp is a sampling-based algorithm (non-deterministic approx. decoding algo.) 
                        # therefore we use multinomial(1) to sample the most likely token from the predicted distribution.
                    
                    if isinstance(model, model_utils.SelfTerminatingLM):
                        output, hidden, p_eos_prev = model.step(xt, hidden, p_eos_prev[:,-1:,:])
                        p_eoss.append(p_eos_prev)
                    else:
                        output, hidden = model.step(xt, hidden, return_all=True)
                    x.append(xt)
                x = torch.cat(x, 1)
            prefixes.append(prefix)
            
            if isinstance(x, torch.Tensor):
                xs.append(x)
            else:
                xs.extend(x)
                
            if isinstance(target, torch.Tensor):
                target = torch.nn.functional.pad(target, (0, max_steps - target.size(1)), "constant", 0)
                targets.append(target)
            else:
                targets.append(target)
            if num_samples >= 0 and (minibatch_id + 1) * inp.size(0) > num_samples:
                break
        if isinstance(xs[0], torch.Tensor):
            xs = torch.cat(xs, 0).tolist()
        
        prefixes = torch.cat(prefixes, 0).tolist()
        if isinstance(targets[0], torch.Tensor):
            targets = torch.cat(targets, 0).tolist()
        
        for i, x in enumerate(xs):
            if eos_token in x:
                xs[i] = x[: x.index(eos_token)]
        for i, x in enumerate(targets):
            if eos_token in x:
                targets[i] = x[: x.index(eos_token)]
        return xs, prefixes, targets


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    consistent_sampling: bool = False,
    eos_idx: int = None,
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        if consistent_sampling:
            indices_to_remove[:, eos_idx].fill_(False)
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        if consistent_sampling:
            indices_to_remove[:, eos_idx].fill_(False)

        logits[indices_to_remove] = filter_value
    return logits

