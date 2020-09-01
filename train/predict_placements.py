# predict step placements + threshold optimization

from scipy.signal import argrelextrema
import numpy as np
import torch
import torch.nn.functional as F

from hyper import PLACEMENT_THRESHOLDS, MAX_CHARTLEVEL, MIN_THRESHOLD

def predict_placements(logits, levels, lengths, get_probs=False, thresholds=None):
    """
    given a sequence of logits from a placement model, predict which positions have steps.
        input:  logits [batch, unroll_length, 2]
                levels [batch] - chart levels used to pick cutoff for a step prediction
                lengths: [batch] for each example, current length
        output: predictions [batch, unroll_length]; 1 -> step, 0 -> no step
    """
    # compute probability dists. for each timestep [batch, unroll, 2]
    probs = F.softmax(logits, dim=-1)

    thresholds_to_use = thresholds if thresholds else PLACEMENT_THRESHOLDS

    # [batch, unroll (lengths)]
    predictions = torch.zeros(logits.size(0), logits.size(1))
    for b in range(logits.size(0)):
        # from https://github.com/chrisdonahue/ddc/blob/master/infer/ddc_server.py
        probs_smoothed = np.convolve(probs[b, :, 1].cpu().numpy(), np.hamming(5), 'same')
        maxima = argrelextrema(probs_smoothed, np.greater_equal, order=1)[0]

        for i in maxima:
            predictions[b, i] = probs[b, i, 1] >= thresholds_to_use[levels[b] - 1]

    if get_probs:
      return predictions, probs[:, :, 1]
    else:
      return predictions

def get_optimal_threshold(placement_model, test_iter, criterion, device):
    placement_model.eval()

    thresholds = {}

    for level in range(MAX_CHARTLEVEL):
        threshold = MIN_THRESHOLD 
        thresholds[level] = threshold
        f2_score = 0

        last_improved = 0

        # find threshold which maximizes the f2 score
        with torch.no_grad():
            # stop optimizing when haven't improved in 5 changes
            while last_improved < 5 and f2_score < 1:
                for batch in testt_iter:
                    pass

    return thresholds

