# predict step placements + threshold optimization

from scipy.signal import argrelextrema
from sklearn.metrics import fbeta_score
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
            predictions[b, i] = probs[b, i, 1] >= thresholds_to_use[str(levels[b])]

    if get_probs:
      return predictions, probs[:, :, 1]
    else:
      return predictions

def get_targets_and_probs(placement_model, valid_iter, device):   
    placement_model.eval()

    # store the targets and prediction scores for the model on the test set (sep. by level)
    all_targets, all_probs = {}, {}
    with torch.no_grad():
        for batch in valid_iter:
            audio_feats = batch['audio_feats'].to(device)
            num_audio_frames = audio_feats.size(2)
            batch_size = audio_feats.size(0)

            audio_lengths = torch.tensor(batch['audio_lengths'], dtype=torch.long, device=device)

            chart_feats = batch['chart_feats'].to(device)
            levels = batch['chart_levels']
            
            targets = batch['placement_targets'].to(device)

            states = placement_model.rnn.initStates(batch_size=batch_size, device=device)

            # [batch, n_audio_frames, 2]
            logits, _, _ = placement_model(audio_feats, chart_feats, states, audio_lengths)

            probs = F.softmax(logits, dim=-1).squeeze(0)

            for b in range(batch_size):
                curr_level = levels[b]
                for i in range(audio_lengths[b]):
                    target = targets[b, i]
                    prob = probs[b, i]

                    if curr_level in all_targets:
                        all_targets[curr_level].append(target.item())
                    else:
                        all_targets[curr_level] = [target.item()]

                    if levels[i] in all_probs:
                        all_probs[curr_level].append(prob.item())
                    else:
                        all_probs[curr_level] = [prob.item()]

    return all_targets, all_probs

def optimize_placement_thresholds(placement_model, valid_iter, device, num_iterations=150):
    thresholds = {}

    targets, probs = get_targets_and_probs(placement_model, valid_iter, device)
    missing_levels = []

    for i in range(MAX_CHARTLEVEL):
        best_threshold = MIN_THRESHOLD
        best_f2_score = 0
        last_improved = 0

        if not targets[i]:
            missing_levels.append(i)

        # find threshold which maximizes the f2 score; do 100 iterrations (max)
        # stop optimizing when haven't improved in the last 5 changes or F2 score cannot go higher
        for j in range(num_iterations):
            if last_improved > num_iterations // 5:
                break

            curr_threshold = j / float(num_iterations)
            curr_preds = [prob > curr_threshold for prob in probs[i]]
            curr_f2_score = fbeta(targets[i], curr_preds, beta=2)

            if curr_f2_score > best_f2_score:
                best_f2_score = curr_f2_score
                last_improved = 0
                thresholds[str(i + 1)] = curr_threshold
            else:
                last_improved += 1


    # default to avg. if no chart examples for a certain level
    for j in missing_levels:
        thresholds[str(j + 1)] = (max(thresholds.values()) - min(thresholds.values())) / 2

    return thresholds
