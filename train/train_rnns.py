# train a group of cnn/rnn models on the same dataset to perform step placement/selection
# (models defined in arrow_rnns.py)

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hyper import *
from arrow_rnns import PlacementCNN, PlacementRNN, SelectionRNN
from stepchart import StepchartDataset, get_splits, collate_charts
from util import report_memory

ABS_PATH = str(Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset under ../data/datasets/subsets/')
    parser.add_argument('--dataset_path', type=str, default=None, help='Alternatively, provide direct path to dataset json file')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help="""Specify custom output directory to save models to. If blank, will save in models/dataset_name/""")
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
    correct = 0
    total_preds = 0
    for b in range(predictions.size(0)):
        for s in range(lengths[b]):
            total_preds += 1

            pred_bs = predictions[b, s].item()
            frame_match = (targets[b, s] == pred_bs).item()
            
            # count placements as positives if within +/- 20 ms (+/- 1 frame)
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

def get_sequence_lengths(curr_unrolling, sequence_unroll_lengths, unroll_length):
    """compute relevant batch metavalues for padding/lengths"""
    sequence_lengths = []
    for b in range(len(sequence_unroll_lengths)):
        padded = curr_unrolling > sequence_unroll_lengths[b][0]
        sequence_lengths.append(sequence_unroll_lengths[b][1] if padded else unroll_length)
    
    return sequence_lengths

def run_placement_batch(cnn, rnn, placement_params, optimizer, criterion, batch, 
                        device, writer, do_train, curr_train_epoch=0):
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

    # [batch, # chart features]
    chart_feats = batch['chart_feats'].to(device)
    #levels = batch['chart_levels'].to(device)

    # [batch, ?] ? = max (audio) sequence length [padded]
    placement_targets = batch['placement_targets'].to(device)

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
                             batch['audio_lengths'][b] % PLACEMENT_UNROLLING_LEN) for b in range(batch_size)]

    total_loss = 0
    total_accuracy = 0

    # for pr curve
    all_targets, all_scores = [], []

    first_frame = batch['first_step_frame']
    last_frame = batch['last_step_frame']

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

        audio_lengths = get_sequence_lengths(unrolling, audio_unroll_lengths, unroll_length)

        # [batch, unroll_len, 83] (batch, unroll length, audio features)
        cnn_input = audio_feats[:, :, audio_start_frame:audio_end_frame]
        cnn_output = cnn(cnn_input)

        # [batch, unroll_len, 2] / [batch, unroll_len, hidden] / ([batch, hidden] x 2)
        logits, clstm_hidden, states = rnn(cnn_output, chart_feats, states, audio_lengths)

        # [batch, unroll_len]
        targets = placement_targets[:, audio_start_frame:audio_end_frame]

        for b in range(batch_size):
            curr_unroll_audio_placements = (targets[b] == 1).nonzero(as_tuple=False).flatten()

            # only clstm hiddens for chart frames with non-empty step placements
            clstm_hiddens[b].append(clstm_hidden[b, curr_unroll_audio_placements])

            all_targets.append(targets[b, :audio_lengths[b]])
            b_dists = F.softmax(logits[b, :audio_lengths[b]], dim=-1)
            all_scores.append(b_dists[:, 1])
        
        # compute total loss for this unrolling
        loss = 0
        for step in range(unroll_length):
            loss += criterion(logits[:, step, :], targets[:, step])

        if do_train:
            loss.backward()
            
            # clip grads before step if L2 norm > 5
            nn.utils.clip_grad_norm_(placement_params, max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
        
        # TODO fix levels
        #predictions = predict_placements(logits, levels, audio_lengths)
        #total_accuracy += get_placement_accuracy(predictions, targets, audio_lengths)

        total_loss += loss.item()

    clstm_hiddens = [torch.cat(hiddens_seq, dim=0) for hiddens_seq in clstm_hiddens]

    if do_train:
        targets = torch.cat(all_targets, dim=0)
        scores = torch.cat(all_scores, dim=0)

        all_targets.clear()
        all_scores.clear()

        writer.add_pr_curve('placement_pr_curve', targets, scores, curr_train_epoch)

    return total_loss / num_audio_frames, total_accuracy / num_audio_frames, clstm_hiddens

def run_selection_batch(rnn, optimizer, criterion, batch, device, clstm_hiddens, do_train):
    if do_train:
        rnn.train()
    else:
        rnn.eval()

    breakpoint()

    # [batch, (max batch) sequence length, chart features] / [batch, (max) seq length]
    step_sequence = batch['step_sequence'].to(device)
    step_targets = batch['step_targets'].to(device)
    batch_size = step_sequence.size(0)

    # lengths of each step sequence; should equal lengths of clstm_hiddens sublists
    step_sequence_lengths = batch['step_sequence_lengths']
    num_frames = max(step_sequence_lengths)

    # note these are unrolling chart frams, vs. audio frames as in placement training
    num_unrollings = (num_frames // SELECTION_UNROLLING_LEN) + 1
    last_unroll_len = (num_frames % SELECTION_UNROLLING_LEN)

    # same as in placement
    seq_unroll_lengths = [(step_sequence_lengths[b] // SELECTION_UNROLLING_LEN,
                           step_sequence_lengths[b] % SELECTION_UNROLLING_LEN) for b in range(batch_size)]

    total_loss = 0
    total_accuracy = 0

    # can use all zeros as start token, since we exclude all empty steps
    start_token = torch.zeros(step_sequence.size(0), 1, step_sequence.size(2), device=device)
    hidden, cell = rnn.initStates(batch_size, device)

    logits, hidden, cell = rnn(start_token, hidden, hidden, cell)
    loss = criterion(logits[:, 0], targets[:, 0:1])

    for unrolling in range(num_unrollings):
        start_frame = unrolling * SELECTION_UNROLLING_LEN
        if unrolling == num_unrollings - 1:
            end_frame = start_frame + last_unroll_len
            last_unrolling = False
        else:
            end_frame = start_frame + SELECTION_UNROLLING_LEN
            last_unrolling = True
        unroll_length = end_frame - start_frame

        if unroll_length == 0:
            continue

        # [batch, unroll_length, # step features]
        step_inputs = step_sequence[:, start_frame:end_frame, :]
        curr_clstm_hiddens = clstm_hiddens[start_frame:end_frame]
        
        curr_seq_lengths = get_sequence_lengths(unrolling, seq_unroll_lengths, unroll_length)

        # [batch, unroll, vocab_size] / [batch, hidden] x 2
        logits, hidden, cell = rnn(step_inputs, curr_clstm_hiddens, hidden, cell, curr_seq_lengths)

        # get targets for next timestep, compute loss; [batch, unroll]
        last_target = end_frame + 1
        targets = step_targets[:, start_frame + 1:end_frame + 1]

        loss = 0
        for step in range(unroll_length):
            # skip final step
            if last_unrolling and step == unroll_length - 1:
                continue

            loss += criterion(logits[:, step, :], targets[:, step])

        if do_train:
            loss.backward()
            
            nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / num_frames, total_accuracy / num_frames

def evaluate(p_cnn, p_rnn, p_params, s_rnn, data_iter, p_criterion, s_criterion,
             device, writer, curr_validation):
    total_p_loss = 0
    total_s_loss = 0
    total_p_acc = 0
    total_s_acc = 0

    with torch.no_grad():
       for i, batch in enumerate(data_iter):
            p_loss, p_acc, hiddens = run_placement_batch(p_cnn, p_rnn, p_params, None,
                p_criterion, batch, device, writer, do_train=False)

            total_p_loss += p_loss
            total_p_acc += p_acc

            s_loss, s_acc = run_selection_batch(s_rnn, None, s_criterion,
                batch, device, hiddens, do_train=False)
            
            total_s_loss += s_loss
            total_s_acc += s_acc

            if curr_validation >= 0:
                step = curr_validation * len(data_iter) + i
                writer.add_scalar('placement_valid_loss', p_loss, step)
                writer.add_scalar('placement_valid_accuracy', p_acc, step)
                writer.add_scalar('selection_valid_loss', s_loss, step)
                writer.add_scalar('selection_valid_accuracy', s_acc, step)

    return (total_p_loss / len(data_iter), total_p_acc / len(data_iter),
            total_s_loss / len(data_iter), total_s_acc / len(data_iter))

# full training process from placement -> selection
def run_models(train_iter, valid_iter, test_iter, num_epochs, dataset_type, device, save_dir, load_checkpoint,
               restart, early_stopping=True, print_every_x_epoch=1, validate_every_x_epoch=2):
    cpu_device = torch.device('cpu')

    # setup  or load models, optimizers
    placement_cnn = PlacementCNN(PLACEMENT_CHANNELS, PLACEMENT_FILTERS, PLACEMENT_KERNEL_SIZES,
                                 PLACEMENT_POOL_KERNEL, PLACEMENT_POOL_STRIDE, device).to(device)
    placement_rnn = PlacementRNN(NUM_PLACEMENT_LSTM_LAYERS, PLACEMENT_INPUT_SIZE, HIDDEN_SIZE).to(device)
    placement_params = list(placement_cnn.parameters()) + list(placement_rnn.parameters())
    placement_optim = optim.SGD(placement_params, lr=PLACEMENT_LR, momentum=0.9)

    selection_rnn = SelectionRNN(NUM_SELECTION_LSTM_LAYERS, SELECTION_INPUT_SIZES[dataset_type], 
                                 SELECTION_VOCAB_SIZES[dataset_type], HIDDEN_SIZE,
                                 SELECTION_HIDDEN_WEIGHT).to(device)
    selection_optim = optim.SGD(selection_rnn.parameters(), lr=SELECTION_LR, momentum=0.9)

    # load model, optimizer states if resuming training
    if load_checkpoint:
        best_placement_valid_loss, best_selection_valid_loss, start_epoch = load_save(load_checkpoint)
    else:
        best_placement_valid_loss = float('inf')
        best_selection_valid_loss = float('inf')
        start_epoch = 0

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs'))

    print('Starting training..')
    for epoch in tqdm(range(num_epochs)):
        if epoch < start_epoch:
            continue

        print('Epoch: {}'.format(epoch))
        epoch_p_loss = 0
        epoch_s_loss = 0
        # if device is gpu: report_memory()

        for i, batch in enumerate(train_iter):
            placement_loss, placement_acc, clstm_hiddens = run_placement_batch(placement_cnn, 
                placement_rnn, placement_params, placement_optim, PLACEMENT_CRITERION,
                batch, device, writer, do_train=True, curr_train_epoch=epoch)

            selection_loss, selection_acc = run_selection_batch(selection_rnn, selection_optim,
                SELECTION_CRITERION, batch, device, clstm_hiddens, do_train=True)

            epoch_p_loss += placement_loss
            epoch_s_loss += selection_loss

            step = epoch * len(train_iter) + i
            writer.add_scalar('placement_train_loss', placement_loss, step)
            writer.add_scalar('placement_train_accuracy', placement_acc, step)
            writer.add_scalar('selection_train_loss', selection_loss, step)
            writer.add_scalar('selection_train_accuracy', selection_acc, step)
        
        epoch_p_loss /= len(train_iter)
        epoch_s_loss /= len(train_iter)

        if epoch % print_every_x_epoch == 0:
            print(f'\tAvg. batch placement loss per timestep: {epoch_p_loss:.3f}')
            print(f'\tAvg. batch selection loss per timestep: {epoch_s_loss:.3f}')

        if epoch % validate_every_x_epoch == 0:
            p_v_loss, p_v_acc, s_v_loss, s_v_acc = evaluate(placement_cnn, placement_rnn, 
                placement_params, selection_rnn, valid_iter, PLACEMENT_CRITERION, 
                SELECTION_CRITERION, device, writer, epoch / validate_every_x_epoch)

            # track best performing model(s)
            if early_stopping:
                if p_v_loss < best_placement_valid_loss:
                    best_placement_valid_loss = p_v_loss
                    save_checkpoint(placement_cnn, placement_rnn, selection_rnn,
                        epoch, best_placement_valid_loss, best_selection_valid_loss,
                        save_dir, best=True)
                else:
                    print("Validation loss increased. Stopping early..")
                    writer.close()
                    breakpoint
        
        if epoch == num_epochs - 1:
            save_checkpoint(placement_cnn, placement_rnn, selection_rnn,
                epoch, best_placement_valid_loss, best_selection_valid_loss,
                save_dir)
    
    writer.close()

    # evaluate on test set
    p_t_loss, p_t_acc, s_t_loss, s_t_acc = evaluate(placement_cnn, placement_rnn, placement_params,
        selection_rnn, test_iter, PLACEMENT_CRITERION, SELECTION_CRITERION, device, writer, -1)

    # save training summary stats to json file
    summary_json = OrderedDict([
        ('epochs_trained', num_epochs),
        ('train_examples', len(train_iter.dataset)),
        ('valid_examples', len(valid_iter.dataset)),
        ('test_examples', len(test_iter.dataset)),
        ('placement_test_loss', p_t_loss),
        ('placement_test_accuracy', p_t_loss),
        ('selection_test_loss', s_t_loss),
        ('selection_test_accuracy', s_t_acc)
    ])

    with open(os.path.join(save_dir, 'summary.json')) as f:
        f.write(json.dumps(summary_json))

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)

    # Retrieve/prepare data
    print('Loading dataset from {}...'.format(os.path.relpath(args.dataset_path)))
    dataset = StepchartDataset(args.dataset_path)
    dataset_type = dataset.chart_type

    train_data, valid_data, test_data = get_splits(dataset)
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_charts, shuffle=True)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_charts, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_charts, shuffle=True)

    datasets_size_str = f"""Total charts in dataset: {len(dataset)}; Train: {len(train_data)}, 
                            Valid: {len(valid_data)}, Test: {len(test_data)}"""
    print(datasets_size_str)

    run_models(train_iter, valid_iter, test_iter, NUM_EPOCHS, dataset_type, device, 
               args.save_dir, args.load_checkpoint, args.restart)

if __name__ == '__main__':
    main()
