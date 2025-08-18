import torch


def compute_metrics(gold_index, predictions):

    hit_1 = 1 if gold_index == predictions[0] else 0
    rank_pos = (predictions == gold_index).nonzero(as_tuple=True)[0] + 1
    mrr = 1 / rank_pos

    return hit_1, mrr[0].item()