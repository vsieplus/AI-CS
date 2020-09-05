# predict step placements + threshold optimization

import argparse
import json
import os
from pathlib import Path

from scipy.signal import argrelextrema
from sklearn.metrics import fbeta_score
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from arrow_rnns import PlacementCLSTM
from hyper import PLACEMENT_THRESHOLDS, MAX_CHARTLEVEL, MIN_THRESHOLD, SUMMARY_SAVE, THRESHOLDS_SAVE
from stepchart import StepchartDataset, collate_charts
from train_util import load_save

ABS_PATH = str(Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
OPTIMIZATION_BATCH_SIZE = 4

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
        for batch in tqdm(valid_iter):
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

            probs = F.softmax(logits.detach(), dim=-1).squeeze(0)

            for b in range(batch_size):
                curr_level = levels[b]
                for i in range(audio_lengths[b]):
                    target = targets[b, i]
                    prob = probs[b, i]

                    if curr_level in all_targets:
                        all_targets[curr_level].append(target.item())
                    else:
                        all_targets[curr_level] = [target.item()]

                    if levels[b] in all_probs:
                        all_probs[curr_level].append(prob.item())
                    else:
                        all_probs[curr_level] = [prob.item()]

    return all_targets, all_probs

def optimize_placement_thresholds(placement_model, valid_iter, device=torch.device('cpu'), num_iterations=300):
    thresholds = {}

    print('Obtaining validation targets and prediction scores')
    targets, probs = get_targets_and_probs(placement_model, valid_iter, device)
    missing_levels = []

    print('Now optimizing thresholds')
    for i in trange(MAX_CHARTLEVEL):
        best_threshold = MIN_THRESHOLD
        best_f2_score = 0
        last_improved = 0

        if not targets[i]:
            missing_levels.append(i)
            continue

        # find threshold which maximizes the f2 score; do given # of iterations (max)
        # stop optimizing when haven't improved in 20% of iterations or reach end
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
        thresholds[str(j + 1)] = sum(thresholds.values()) / len(thresholds)

    return thresholds

# run the script directly to optimize placement thresholds for an already trained model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Directory with model files')
    parser.add_argument('--dataset_name', type=str, help='Name of dataset')

    args = parser.parse_args()

    device = torch.device('cpu') 

    print(f'Loading dataset {args.dataset_name}')

    dataset_path = os.path.join(DATASETS_DIR, args.dataset_name + '.json')
    dataset = StepchartDataset(dataset_path, load_to_memory=False, first_dataset_load=False, special_tokens=None) 

    _, valid_indices, _ = dataset.get_splits()
    valid_iter = DataLoader(dataset, batch_size=OPTIMIZATION_BATCH_SIZE, collate_fn=collate_charts, sampler = SubsetRandomSampler(valid_indices))

    print('Loading model files')
    with open(os.path.join(args.model_dir, SUMMARY_SAVE), 'r') as f:
        model_summary = json.loads(f.read())

    placement_model = PlacementCLSTM(model_summary['placement_channels'], model_summary['placement_filters'], 
                                     model_summary['placement_kernels'], model_summary['placement_pool_kernel'],
                                     model_summary['placement_pool_stride'], model_summary['placement_lstm_layers'],
                                     model_summary['placement_input_size'], model_summary['hidden_size']).to(device)
    load_save(args.model_dir, fine_tune=True, placement_clstm=placement_model, selection_rnn=None, device=device)

    print('Optimizing thresholds')
    thresholds = optimize_placement_thresholds(placement_model, valid_iter, device)

    with open(os.path.join(args.model_dir, THRESHOLDS_SAVE), 'w') as f:
        f.write(json.dumps(thresholds, indent=2))

    print(f'Thresholds saved to {os.path.join(args.model_dir, THRESHOLDS_SAVE)}')
