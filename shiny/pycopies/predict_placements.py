# predict step placements

from scipy.signal import argrelextrema
import numpy as np
import torch
import torch.nn.functional as F

from hyper import PLACEMENT_THRESHOLDS

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
            predictions[b, i] = probs[b, i, 1] >= thresholds_to_use[str(levels[b])]

    if get_probs:
      return predictions, probs[:, :, 1]
    else:
      return predictions