# train a group of cnn/rnn models on the same dataset to perform step placement/selection
# (models defined in arrow_rnns.py)

import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from hyper import *
from arrow_rnns import PlacementCNN, PlacementRNN, SelectionRNN
from stepchart import StepchartDataset, get_splits, collate_charts
from util import report_memory

ABS_PATH = str(Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset under data/datasets/subsets/')
    parser.add_argument('--dataset_path', type=str, default=None, help='Alternatively, provide direct path to dataset json file')
    parser.add_argument('--save_dir', type=str, default=None, help="""Specify custom output directory to save
        models to. If blank, will save in models/dataset_name/""")
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load models from the specified checkpoint')
    parser.add_argument('--restart', action='store_true', default=False,
        help='Use this option to start training from beginning with a checkpoint; Else resume training from when stopped')

    args = parser.parse_args()

    if not args.dataset_path:
        if args.dataset_name:
            args.dataset_path = os.path.join(DATASETS_DIR, args.dataset_name + '.json')
        elif not args.load_checkpoint:
            raise ValueError('--dataset_name, --dataset_path, or --load_checkpoint required')

    if not args.save_dir:
        args.save_dir = os.path.join(MODELS_DIR, os.path.split(args.dataset_path)[-1].split('.')[0])

    return args

def save_checkpoint(p_cnn, p_rnn, s_rnn, epoch, best_p_vloss, best_s_vloss, save_dir, best=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    out_path = os.path.join(save_dir, 'checkpoint.tar')
    
    print(f'\tSaving checkpoint to {out_path}')
    torch.save({
        'epoch': epoch,
        'best_placement_valid_loss': best_p_vloss,
        'best_selection_valid_loss': best_s_vloss,
        'p_cnn_state_dict': p_cnn.state_dict(),
        'p_rnn_state_dict': p_rnn.state_dict(),
        's_rnn_state_dict': s_rnn.state_dict(),
    }, out_path)

    if best:
        shutil.copy(out_path, os.path.join(save_dir, 'model_best.tar'))

def load_save(path):
    print(f'Loading checkpoint from {path}...')
    checkpoint = torch.load(path)

    # only restore epoch/best loss values when not restarting    
    # (i.e. give option to restart training on a pretrained-model)       
    if not restart:
        start_epoch = checkpoint['epoch']
        best_placement_valid_loss = checkpoint['best_placement_valid_loss']
        best_selection_valid_loss = checkpoint['best_selection_valid_loss']

    placement_cnn.load_state_dict(checkpoint['p_cnn_state_dict'])
    placement_rnn.load_state_dict(checkpoint['p_rnn_state_dict'])
    selection_rnn.load_state_dict(checkpoint['s_rnn_state_dict'])

    return start_epoch, best_placement_valid_loss, best_selection_valid_loss

def predict_placements(logits, levels, lengths):
    """given a sequence of logits from a placement model, predict which positions have steps.
        input:  logits [batch, unroll_length, 2]
                levels [batch] - chart levels used to pick cutoff for a step prediction
                lengths: [batch] for each example, current length
        output: predictions [batch, unroll_length]; 1 -> step, 0 -> no step
    """

    thresholds = MAX_THRESHOLD - ((torch.tensor(levels, dtype=torch.float) - 1) / 
        (MAX_CHARTLEVEL - 1)) * (MAX_THRESHOLD - MIN_THRESHOLD)

    # compute probability dists. for each timestep [batch, unroll, 2]
    probs = F.softmax(logits, dim=-1)
    #hamming_window = torch.hamming_window(window_length=probs.size(1))

    # [batch, unroll (lengths)]
    predictions = torch.zeros(logits.size(0), logits.size(1))
    for b in range(logits.size(0)):
        predictions[b, :lengths[b]] = probs[b, :lengths[b],  1] >= thresholds[b]

    return predictions

def get_placement_accuracy(predictions, targets, lengths):
    """get placement accuracy between given predictions/targets"""

    # TODO add precision/recall

    correct = 0
    total_preds = 0
    for b in range(predictions.size(0)):
        for s in range(lengths[b]):
            total_preds += 1

            pred_bs = predictions[b, s].item()
            frame_match = (targets[b, s] == pred_bs).item()
            
            # count a placements as positives if within +/- 20 ms (+/- 1 frame)
            # (assume no placements at the ends)
            if s == 0:
                shift_match = (targets[b, s + 1] == pred_bs).item()
            elif s == lengths[b] - 1:
                shift_match = (targets[b, s - 1] == pred_bs).item()
            else:
                shift_match = (targets[b, s - 1] == pred_bs or 
                               targets[b, s + 1] == pred_bs).item()

            correct += frame_match or shift_match

    return correct / total_preds

def get_step_placement_frames(chart_placements, step_frames, batch_size):
    # the indices of step_frames which had step placements
    step_placement_indices = [(chart_placements[b] == 1).nonzero(as_tuple=False).squeeze().tolist()
                              for b in range(batch_size)]

    # the subset of step_frames (audio frames) that had placements
    step_placement_frames = torch.tensor([[step_frames[b][idx] for idx in step_placement_indices[b]]
                                          for b in range(batch_size)])
                                           
    # compute the first and last audio frames coinciding with chart non-emtpy step placements
    first_frame = min(step_frames[b][indices[0]] for b, indices in enumerate(step_placement_indices))
    last_frame = max(step_frames[b][indices[-1]] for b, indices in enumerate(step_placement_indices))

    return step_placement_frames, first_frame, last_frame

def run_placement_batch(cnn, rnn, placement_params, optimizer, criterion, batch, device, do_train):
    """train or eval the placement models with the specified parameters on the given batch"""
    if do_train:
        cnn.train()
        rnn.train()
    else:
        cnn.eval()
        rnn.eval()

    # [batch, 3, ?, 80] ; ? = (batch maximum) # of timesteps/frames in audio features
    audio_feats = batch['audio_feats'].to(device)
    num_audio_frames = audio_feats.size(2)
    batch_size = audio_feats.size(0)

    sample_rate = batch['sample_rate']

    # [batch, # chart features]
    chart_feats = batch['chart_feats'].to(device)
    levels = []
    for b in range(batch_size):
        levels.append((chart_feats[b, 2:] == 1).nonzero(as_tuple=False).flatten().item() + 1)

    # both are [batch, max # of chart frames] (padded/not padded)
    chart_placements = batch['placement_targets']   # if chart frame had a hit or not
    step_frames = batch['step_frames']              # which audio frames recorded in chart data

    # get audio frame numbers of step placements
    step_placement_frames, first_frame, last_frame = get_step_placement_frames(chart_placements, step_frames, batch_size)

    # final_shape -> [batch, # of nonempty chart frames, hidden]
    clstm_hiddens = [[] for _ in range(batch_size)]

    # for unrolling/bptt
    num_unrollings = (num_audio_frames // PLACEMENT_UNROLLING_LEN) + 1
    last_unroll_len = num_audio_frames % PLACEMENT_UNROLLING_LEN
    states = rnn.initStates(batch_size, device)
    
    # get representation of audio lengths for each unrolling
    # [full frames, last frame length]
    # (e.g. [100, 100, 100, 64] -> (3,64))
    audio_unroll_lengths = [(batch['audio_lengths'][b] // PLACEMENT_UNROLLING_LEN, 
                             batch['audio_lengths'][b] % PLACEMENT_UNROLLING_LEN) 
                            for b in range(batch_size)]

    total_loss = 0
    total_accuracy = 0

    # loop through all unrollings
    for unrolling in range(num_unrollings):
        audio_start_frame = unrolling * PLACEMENT_UNROLLING_LEN
        if unrolling == num_unrollings - 1:
            audio_end_frame = audio_start_frame + last_unroll_len
        else:
            audio_end_frame = audio_start_frame + PLACEMENT_UNROLLING_LEN
        unroll_length = audio_end_frame - audio_start_frame

        # skip audio frames outside chart range
        if (audio_end_frame < first_frame or audio_start_frame > last_frame + 1 or unroll_length == 0):
            continue
        
        # convert audio frame #s [from mel spectrogram representation] -> chart frame #s (10ms)
        # MEL_FRAME = round((SAMPLE_RATE * TIME [secs]) / HOP_LENGTH [mel])
        # -> secs = (hop * melframe) / sr
        chart_start_frame = round(audio_start_frame * HOP_LENGTH / sample_rate * CHART_FRAME_RATE)
        chart_end_frame = round(audio_end_frame * HOP_LENGTH / sample_rate * CHART_FRAME_RATE)

        # compute relevant batch metavalues for padding/lengths
        padded_seqs = [unrolling > audio_unroll_lengths[b][0] - 1 for b in range(batch_size)]
        audio_lengths = [audio_unroll_lengths[b][1] if padded_seqs[b] else unroll_length
                         for b in range(batch_size)]

        # [batch, unroll_len, 83] (batch, unroll length, audio features)
        cnn_input = audio_feats[:, :, audio_start_frame:audio_end_frame]
        cnn_output = cnn(cnn_input)

        # [batch, unroll_len, 2] / [batch, unroll_len, hidden] / (*...hidden x 2)
        logits, clstm_hidden, states = rnn(cnn_output, chart_feats, states, audio_lengths)

        # for each batch example, if this audio frame also in the chart, use actual target at that frame
        # otherwise use default zero value for no placement at all
        targets = torch.zeros(batch_size, unroll_length, dtype=torch.long, device=device)
        for b in range(batch_size):
            # which frames to consider for the current unrolling
            curr_unroll_chart_frames = torch.logical_and(step_placement_frames[b] >= chart_start_frame,
                                                         step_placement_frames[b] < chart_end_frame)

            # which of them had placements?
            curr_unroll_chart_placements = step_placement_frames[b][curr_unroll_chart_frames]

            # convert to respective audio frames, + normalize to [0, unroll_length-1]
            curr_unroll_audio_placements = (curr_unroll_chart_placements / CHART_FRAME_RATE 
                * SAMPLE_RATE / HOP_LENGTH).int() - audio_start_frame

            targets[b, curr_unroll_audio_placements] = 1

            # for padded seqs, set targets to the ignore_index val.
            targets[b, audio_lengths[b]:] = PAD_IDX

            # only clstm hiddens for chart frames with non-empty step placements
            clstm_hiddens[b].append(clstm_hidden[b, curr_unroll_audio_placements])

        # compute total loss for this unrolling
        loss = 0
        for step in range(unroll_length):
            loss += criterion(logits[:, step, :], targets[:, step])

        if do_train:
            loss.backward()
            
            # clip grads before step if L2 norm > 5
            nn.utils.clip_grad_norm_(placement_params, max_norm=PLACEMENT_MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()
        
        # compute accuracy
        predictions = predict_placements(logits, levels, audio_lengths)
        total_accuracy += get_placement_accuracy(predictions, targets, audio_lengths)

        total_loss += loss.item()

    clstm_hiddens = [torch.cat(hiddens_seq, dim=0) for hiddens_seq in clstm_hiddens]

    return total_loss / num_audio_frames, total_accuracy / num_audio_frames, clstm_hiddens

def run_selection_batch(rnn, optimizer, criterion, batch, device, clstm_hiddens, do_train):
    if do_train:
        rnn.train()
    else:
        rnn.eval()

    breakpoint()

    step_sequence = batch['step_sequence']

    # lengths of each step sequence; should equal lengths of clstm_hiddens sublists
    sequence_lengths = batch['sequence_lengths']
    num_frames = max(sequence_lengths)

    num_unrollings = (num_frames // SELECTION_UNROLLING_LEN) + 1
    last_unroll_len = (num_frames % SELECTION_UNROLLING_LEN)

    total_loss = 0
    total_accuracy = 0

    for unrolling in range(num_unrollings):
        start_frame = unrolling * SELECTION_UNROLLING_LEN
        if unrolling == num_unrollings - 1:
            end_frame = start_frame + last_unroll_len
        else:
            audio_end_frame = start_frame + SELECTION_UNROLLING_LEN
        unroll_length = end_frame - start_frame

    return total_loss / num_frames, total_accuracy / num_frames

def evaluate(p_cnn, p_rnn, p_params, s_rnn, valid_iter, p_criterion, s_criterion, device):
    total_p_loss = 0
    total_s_loss = 0
    total_p_acc = 0
    total_s_acc = 0

    with torch.no_grad():
       for batch in valid_iter:
            p_loss, p_acc, hiddens = run_placement_batch(p_cnn, p_rnn, p_params, None,
                p_criterion, batch, device, do_train=False)

            total_p_loss += p_loss
            total_p_acc += p_acc

            s_loss, s_acc = run_selection_batch(s_rnn, None, s_criterion,
                batch, device, hiddens, do_train=False)
            
            total_s_loss += s_loss
            total_s_acc += s_acc

    return (total_p_loss / len(valid_iter), total_p_acc / len(valid_iter),
            total_s_loss / len(valid_iter), total_s_acc / len(valid_iter))

# full training process from placement -> selection
def run_models(train_iter, valid_iter, num_epochs, dataset_type, device, save_dir, load_checkpoint,
               restart, early_stopping=True, print_every_x_epoch=1, validate_every_x_epoch=2):
    cpu_device = torch.device('cpu')

    # setup  or load models, optimizers
    placement_cnn = PlacementCNN(PLACEMENT_CHANNELS, PLACEMENT_FILTERS, PLACEMENT_KERNEL_SIZES,
                                 PLACEMENT_POOL_KERNEL, PLACEMENT_POOL_STRIDE, device).to(device)
    placement_rnn = PlacementRNN(NUM_PLACEMENT_LSTM_LAYERS, PLACEMENT_INPUT_SIZE, HIDDEN_SIZE).to(device)
    placement_params = list(placement_cnn.parameters()) + list(placement_rnn.parameters())
    placement_optim = optim.SGD(placement_params, lr=PLACEMENT_LR, momentum=0.9)

    selection_rnn = SelectionRNN(NUM_SELECTION_LSTM_LAYERS, SELECTION_VOCAB_SIZES[dataset_type], 
                                 HIDDEN_SIZE, SELECTION_HIDDEN_WEIGHT).to(device)
    selection_optim = optim.SGD(selection_rnn.parameters(), lr=SELECTION_LR, momentum=0.9)

    # load model, optimizer states if resuming training
    if load_checkpoint:
        best_placement_valid_loss, best_selection_valid_loss, start_epoch = load_save(load_checkpoint)
    else:
        best_placement_valid_loss = float('inf')
        best_selection_valid_loss = float('inf')
        start_epoch = 0

    print('Starting training..')
    for epoch in tqdm(range(num_epochs)):
        if epoch < start_epoch:
            continue

        print('Epoch: {}'.format(epoch))
        epoch_p_loss = 0
        epoch_s_loss = 0
        # if device is gpu: report_memory()

        for i, batch in enumerate(train_iter):
            placement_loss, _, clstm_hiddens = run_placement_batch(placement_cnn, placement_rnn,
                placement_params, placement_optim, PLACEMENT_CRITERION, batch, device, do_train=True)

            selection_loss, _ = run_selection_batch(selection_rnn, selection_optim,
                SELECTION_CRITERION, batch, device, clstm_hiddens, do_train=True)

            epoch_p_loss += placement_loss
            epoch_s_loss += selection_loss
        
        epoch_p_loss /= len(train_iter)
        epoch_s_loss /= len(train_iter)

        if epoch % print_every_x_epoch == 0:
            print(f'\Epoch {epoch}')
            print(f'\tAvg. placement loss per timestep: {epoch_p_loss:.3f}')
            print(f'\tAvg. selection loss per timestep: {epoch_s_loss:.3f}')

        if epoch % validate_every_x_epoch == 0:
            p_v_loss, p_v_acc, s_v_loss, s_v_acc = evaluate(placement_cnn, placement_rnn, placement_params,
                selection_rnn, valid_iter, PLACEMENT_CRITERION, SELECTION_CRITERION, device)

        # track best performing model(s)
        if early_stopping:
            if p_v_loss < best_placement_valid_loss:
                best_placement_valid_loss = p_v_loss
                save_checkpoint(placement_cnn, placement_rnn, selection_rnn,
                    epoch, best_placement_valid_loss, best_selection_valid_loss,
                    save_dir, best=True)
            else:
                print("Validation loss increased. Stopping early..")
                return

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)

    # Retrieve/prepare data
    print('Loading dataset from {}...'.format(os.path.relpath(args.dataset_path)))
    dataset = StepchartDataset(args.dataset_path)
    dataset_type = dataset.chart_type

    train_data, valid_data, test_data = get_splits(dataset)
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)

    print(f"""Total charts in dataset: {len(dataset)}; Train: {len(train_data)}, 
        Valid: {len(valid_data)}, Test: {len(test_data)}""")

    run_models(train_iter, valid_iter, NUM_EPOCHS, dataset_type, device, args.save_dir, 
        args.load_checkpoint, args.restart)

if __name__ == '__main__':
    main()
