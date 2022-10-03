import torch
import numpy as np
import editdistance
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import ngrams
from collections import Counter, defaultdict


class GenerationMetrics(object):
    def __init__(self, distances=('edit', 'sentence_bleu', 'len_diff', 'nonterm', 'repeat-4')):
        self.distances = distances
        self._stats_cache = defaultdict(list)
    
    def step(self, preds_trim, targets_trim, outputs, eos_token, context_length):
        with torch.no_grad():
            for distance in self.distances:
                dist = task_distance(
                    target_trim=targets_trim,
                    model_trim=preds_trim,
                    outputs=outputs,
                    kind=distance,
                    eos_id=eos_token
                )
                self._stats_cache['distance-%s' % distance].append(dist)

            len_diffs, target_lens, model_lens = len_stats(
                preds_trim, targets_trim, eos_token
            )
            self._stats_cache['len_diff'].extend(len_diffs)
            self._stats_cache['target/len'].extend(target_lens)
            self._stats_cache['model/len'].extend(model_lens)
            self._stats_cache['non_term'].extend(
                nonterm_metrics(preds_trim, eos_token)
            )
            for k, vs in ngram_metrics(targets_trim, eos_token).items():
                self._stats_cache['target/%s' % k].extend(vs)
            for k, vs in ngram_metrics(preds_trim, eos_token).items():
                self._stats_cache['model/%s' % k].extend(vs)
    
    def normalize(self, prefix='valid/decode'):
        output = {}
        for key in self._stats_cache:
            output['%s/%s' % (prefix, key)] = np.mean(self._stats_cache[key])
        return output


def task_distance(target_trim, model_trim, outputs, kind='edit', eos_id=None, bleu_smoothing='method2'):
    if kind == 'edit':
        edits = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            edit_dist = editdistance.eval(actual_, predicted_)
            edit_dist = edit_dist / max(len(predicted_), len(actual_))
            edits.append(edit_dist)
        distance = sum(edits) / len(edits)
    elif kind == 'sentence_bleu':
        bleus = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            smoothingf = getattr(SmoothingFunction(), bleu_smoothing)
            bleu = sentence_bleu([actual_], predicted_, smoothing_function=smoothingf)
            bleus.append(bleu)
        distance = 1 - (sum(bleus) / len(bleus))
    elif kind == 'bleu':
        smoothingf = getattr(SmoothingFunction(), bleu_smoothing)
        target_trim_ = [[x] for x in target_trim]
        bleu = corpus_bleu(target_trim_, model_trim, smoothing_function=smoothingf)
        distance = 1 - bleu
    elif kind == 'len_diff':
        diffs = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            diff = np.abs(len(predicted_) - len(actual_))
            diff = diff / max(len(predicted_), len(actual_))
            diffs.append(diff)
        distance = sum(diffs) / len(diffs)
    elif kind == 'nonterm':
        diffs = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            if eos_id in predicted_:
                diffs.append(0.0)
            else:
                diffs.append(1.0)
        distance = sum(diffs) / len(diffs)
    elif kind == 'repeat-4':
        distances = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            if len(predicted_) >= 4:
                ngs = [ng for ng in ngrams(predicted_, 4)]
                counter = Counter(ngs)
                distances.append(1.0 - len(counter)/max(len(ngs), 1))
            else:
                distances.append(1.0)
        distance = np.mean(distances)
    else:
        raise NotImplementedError(kind)
    return distance


def len_stats(decoded, target, eos_token):
    if not isinstance(target, list):
        target = target.tolist()
    if not isinstance(decoded, list):
        decoded = decoded.tolist()
    diffs = []
    target_lens = []
    model_lens = []

    for data_cont, model_cont in zip(target, decoded):
        if eos_token in data_cont:
            data_cont_ = data_cont[:data_cont.index(eos_token)+1]
        else:
            data_cont_ = data_cont
        if eos_token in model_cont:
            model_cont_= model_cont[:model_cont.index(eos_token)+1]
        else:
            model_cont_ = model_cont

        diff = np.abs(len(data_cont_) - len(model_cont_))
        diffs.append(diff)
        target_lens.append(len(data_cont_))
        model_lens.append(len(model_cont_))
    return diffs, target_lens, model_lens


def ngram_metrics(sequences, eos_token):
    stats = defaultdict(list)
    if not isinstance(sequences, list):
        sequences = sequences.tolist()

    for sequence in sequences:
        if eos_token in sequence:
            sequence_ = sequence[:sequence.index(eos_token)+1]
        else:
            sequence_ = sequence

        for n in [1, 4]:
            if len(sequence_) >= n:
                ngs = [ng for ng in ngrams(sequence_, n)]
                counter = Counter([ng for ng in ngrams(sequence_, n)])
                stats['pct_repeat_%dgrams' % n].append(
                    1.0 - len(counter)/max(len(ngs), 1)
                )
    return stats


def nonterm_metrics(sequences, eos_token):
    nonterm = []
    if not isinstance(sequences, list):
        sequences = sequences.tolist()
    for sequence in sequences:
        nonterm.append(float(eos_token not in sequence))
    return nonterm
