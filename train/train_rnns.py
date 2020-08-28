# train a group of cnn/rnn models on the same dataset to perform step placement/selection
# (models defined in arrow_rnns.py)

import argparse
import datetime
import gc
import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

from hyper import *
from arrow_rnns import PlacementCLSTM, SelectionRNN
from stepchart import StepchartDataset, get_splits, collate_charts
from util import report_memory, SummaryWriter

ABS_PATH = str(Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset under ../data/datasets/subsets/')
    parser.add_argument('--dataset_path', type=str, default=None, help='Alternatively, provide direct path to dataset json file')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help="""Specify custom output directory to save models to. If blank, will save in ./models/dataset_name/""")
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load models from the specified checkpoint')
    parser.add_argument('--retrain', action='store_true', default=False, help=('Use this option to (re)train a model that'
        'has already been trained starting from default epoch/validation loss; Otherwise resume training from when stopped'))

    args = parser.parse_args()

    if not args.dataset_path:
        if args.dataset_name:
            args.dataset_path = os.path.join(DATASETS_DIR, args.dataset_name + '.json')
        elif args.load_checkpoint:
            args.dataset_path = os.path.join(DATASETS_DIR, args.load_checkpoint.split(os.sep)[-1] + '.json')
        else:    
            raise ValueError('--dataset_name, --dataset_path, or --load_checkpoint required')

    if not args.save_dir:
        args.save_dir = os.path.join(MODELS_DIR, os.path.split(args.dataset_path)[-1].split('.')[0])

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args

def save_checkpoint(epoch, curr_epoch_batch, best_placement_valid_loss, 
                    best_selection_valid_loss, train_clstm, train_srnn, save_dir):
    out_path = os.path.join(save_dir, CHECKPOINT_SAVE)
    
    print(f'\tSaving checkpoint to {out_path}')
    torch.save({
        'epoch': epoch,
        'curr_epoch_batch': curr_epoch_batch,
        'best_placement_valid_loss': best_placement_valid_loss,
        'best_selection_valid_loss': best_selection_valid_loss,
        'train_clstm': train_clstm,
        'train_srnn': train_srnn,
    }, out_path)

def save_model(model, save_dir, model_filename):
    out_path = os.path.join(save_dir, model_filename)
    print(f'\tSaving model to {out_path}')

    torch.save(model.state_dict(), out_path)

def load_save(save_dir, retrain, placement_clstm, selection_rnn, device):
    print(f'Loading checkpoint from {save_dir}...')

    clstm_path = os.path.join(save_dir, CLSTM_SAVE)
    srnn_path = os.path.join(save_dir, SRNN_SAVE)
    
    if os.path.isfile(clstm_path):
        placement_clstm.load_state_dict(torch.load(clstm_path, map_location=device))
    else:
        print(f'No saved {CLSTM_SAVE} file found in {save_dir}, starting from base model')

    if os.path.isfile(srnn_path):
        selection_rnn.load_state_dict(torch.load(srnn_path, map_location=device))
    else:
        print(f'No saved {SRNN_SAVE} file found in {save_dir}, starting from base model')

    checkpoint = torch.load(os.path.join(save_dir, CHECKPOINT_SAVE))

    # only restore epoch/best loss values when not retraining    
    # (i.e. give option to retrain some more on an already trained model)       
    if not retrain:
        start_epoch = checkpoint['epoch']
        start_epoch_batch = checkpoint['curr_epoch_batch']
        best_placement_valid_loss = checkpoint['best_placement_valid_loss']
        best_selection_valid_loss = checkpoint['best_selection_valid_loss']
        train_clstm = checkpoint['train_clstm']
        train_srnn = checkpoint['train_srnn']

        # use last directory under save_dir/runs
        logdirs = [d for d in os.listdir(os.path.join(save_dir, 'runs'))]
        sub_logdir = [d for d in logdirs if os.path.isdir(os.path.join(save_dir, 'runs', d))][-1]

        return (start_epoch, start_epoch_batch, best_placement_valid_loss, 
                best_selection_valid_loss, train_clstm, train_srnn, sub_logdir)
    else:
        return None

def predict_placements(logits, levels, lengths):
    """given a sequence of logits from a placement model, predict which positions have steps.
        input:  logits [batch, unroll_length, 2]
                levels [batch] - chart levels used to pick cutoff for a step prediction
                lengths: [batch] for each example, current length
        output: predictions [batch, unroll_length]; 1 -> step, 0 -> no step
    """
    # compute probability dists. for each timestep [batch, unroll, 2]
    probs = F.softmax(logits, dim=-1)

    # [batch, unroll (lengths)]
    predictions = torch.zeros(logits.size(0), logits.size(1))
    for b in range(logits.size(0)):
        # from https://github.com/chrisdonahue/ddc/blob/master/infer/ddc_server.py
        probs_smoothed = np.convolve(probs[b].numpy(), np.hamming(5), 'same')
        maxima = argrelextrema(probs_smoothed, np.greater_equal, order=1)[0]

        for i in maxima:
            predictions[b, i] = probs[b, i] >= PLACEMENT_THRESHOLDS[levels[b]]

        # predictions[b, :lengths[b]] = probs[b, :lengths[b],  1] >= PLACEMENT_THRESHOLDS[levels[b]]

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

    return correct / total_preds if total_preds != 0 else 0

def get_sequence_lengths(curr_unrolling, sequence_unroll_lengths, unroll_length, device):
    """compute relevant batch metavalues for padding/lengths"""
    sequence_lengths = []
    for b in range(len(sequence_unroll_lengths)):
        if curr_unrolling == sequence_unroll_lengths[b][0]:
            length = sequence_unroll_lengths[b][1]
        elif curr_unrolling > sequence_unroll_lengths[b][0]:
            length = 0
        else:
            length = unroll_length
        
        sequence_lengths.append(length)
    
    return torch.tensor(sequence_lengths, dtype=torch.long, device=device)

def run_placement_batch(clstm, optimizer, criterion, batch, device, writer, do_train, curr_train_epoch=0):
    """train or eval the placement models with the specified parameters on the given batch"""
    if do_train:
        clstm.train()
    else:
        clstm.eval()

    # [batch, 3, ?, 80] ; ? = (batch maximum) # of timesteps/frames in audio features
    audio_feats = batch['audio_feats'].to(device)
    num_audio_frames = audio_feats.size(2)
    batch_size = audio_feats.size(0)

    # [batch, # chart features]
    chart_feats = batch['chart_feats'].to(device)
    levels = batch['chart_levels']

    # [batch, ?] ? = max (audio) sequence length [padded]
    placement_targets = batch['placement_targets'].to(device)

    # final_shape -> [batch, # of nonempty chart frames, hidden]
    clstm_hiddens = [[] for _ in range(batch_size)]

    # for unrolling/bptt
    num_unrollings = (num_audio_frames // PLACEMENT_UNROLLING_LEN) + 1
    last_unroll_len = num_audio_frames % PLACEMENT_UNROLLING_LEN
    states = clstm.rnn.initStates(batch_size, device)
    
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

        audio_lengths = get_sequence_lengths(unrolling, audio_unroll_lengths, unroll_length, device)

        audio_input = audio_feats[:, :, audio_start_frame:audio_end_frame]

        # [batch, unroll_len, 2] / [batch, unroll_len, hidden] / ([num_lstm_layers, batch, hidden] x 2)
        logits, clstm_hidden, states = clstm(audio_input, chart_feats, states, audio_lengths)

        # [batch, unroll_len]
        targets = placement_targets[:, audio_start_frame:audio_end_frame]

        for b in range(batch_size):
            if audio_lengths[b] == 0:
                continue

            if do_train:
                all_targets.append(targets[b, :audio_lengths[b]])
                b_dists = F.softmax(logits.detach()[b, :audio_lengths[b]], dim=-1)
                all_scores.append(b_dists[:, 1])
            
            curr_unroll_placements = (targets[b] == 1).nonzero(as_tuple=False).flatten()

            # only clstm hiddens for chart frames with non-empty step placements
            clstm_hiddens[b].append(clstm_hidden[b, curr_unroll_placements])
        
        # compute total loss for this unrolling
        loss = 0
        for step in range(logits.size(1)):
            loss += criterion(logits[:, step, :], targets[:, step])

        if do_train:
            loss.backward()
            
            # clip grads before step
            nn.utils.clip_grad_norm_(clstm.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        predictions = predict_placements(logits.detach(), levels, audio_lengths)
        total_accuracy += get_placement_accuracy(predictions, targets, audio_lengths)
        total_loss += loss.item()

    # pad clstm_hiddens across batch examples
    # [batch, [step_sequence_length(variable), hidden]] -> [batch, max_step_sequence_length, hidden]
    clstm_hiddens = [torch.cat(hiddens_seq, dim=0) for hiddens_seq in clstm_hiddens]
    clstm_hiddens_padded = pad_sequence(clstm_hiddens, batch_first=True, padding_value=0)
    clstm_hiddens.clear()

    if do_train:
        targets = torch.cat(all_targets, dim=0)
        scores = torch.cat(all_scores, dim=0)

        all_targets.clear()
        all_scores.clear()

        writer.add_pr_curve('placement_pr_curve', targets, scores, curr_train_epoch)

    return total_loss / num_unrollings, total_accuracy / num_unrollings, clstm_hiddens_padded

def get_selection_accuracy(logits, targets, seq_lengths, step):
    # [batch, vocab_size]
    dists = F.softmax(logits, dim=-1)
    
    # [batch]
    preds = torch.topk(dists, k=1, dim=-1)[1]

    correct, total_preds = 0, 0
    for b, seq_length in enumerate(seq_lengths):
        if seq_length > step:
            total_preds += 1

            correct += (preds[b] == targets[b]).item()
    
    return correct / total_preds if total_preds != 0 else 0

def run_selection_batch(rnn, optimizer, criterion, batch, device, clstm_hiddens, do_train):
    if do_train:
        rnn.train()
    else:
        rnn.eval()
        
    # [batch, (max batch) sequence length, chart features] / [batch, (max) seq length]
    step_sequence = batch['step_sequence'].to(device)
    step_targets = batch['step_targets'].to(device)
    batch_size = step_sequence.size(0)

    # lengths of each step sequence; should equal true sizes of original clstm_hiddens tensors
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
    start_token = [torch.zeros(1, step_sequence.size(2), device=device) for _ in range(step_sequence.size(0))]
    start_token = pad_sequence(start_token, batch_first=True, padding_value=PAD_IDX)
    hidden, cell = rnn.initStates(batch_size, device)

    logits, (hidden, cell) = rnn(start_token, None, hidden, cell, torch.ones(batch_size, dtype=torch.long, device=device))
    loss = criterion(logits[:, 0], step_targets[:, 0])
    if do_train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    total_loss += loss.item()

    for unrolling in range(num_unrollings):
        last_unrolling = unrolling == num_unrollings - 1
        
        start_frame = unrolling * SELECTION_UNROLLING_LEN
        if last_unrolling:
            end_frame = start_frame + last_unroll_len
        else:
            end_frame = start_frame + SELECTION_UNROLLING_LEN
        unroll_length = end_frame - start_frame

        if unroll_length == 0:
            continue

        # total lengths for each sequence for this unrolling
        curr_seq_lengths = get_sequence_lengths(unrolling, seq_unroll_lengths, unroll_length, device)

        # use targets for next timestep, compute loss; [batch, unroll] (already padded)
        loss = 0
        for step in range(unroll_length):
            abs_step = start_frame + step

            # skip final final step
            if abs_step == num_frames - 1:
                break
                
            # [batch, 1, # step features] / [batch, hiddens] - feed clstm hiddens 1 by 1
            step_inputs = step_sequence[:, abs_step, :].unsqueeze(1)
            
            curr_clstm_hiddens = clstm_hiddens[:, abs_step]

            # ones or zeros
            curr_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
            curr_lengths[curr_seq_lengths <= step] = 0
            
            # [batch, 1, vocab_size] / [num_lstm_layers, batch, hidden] x 2
            logits, (hidden, cell) = rnn(step_inputs, curr_clstm_hiddens, hidden, cell, curr_lengths)

            loss += criterion(logits[:, 0], step_targets[:, abs_step + 1])

            total_accuracy += get_selection_accuracy(logits[:, 0], step_targets[:, abs_step + 1], 
                                                     curr_seq_lengths, step)

        if do_train:
            loss.backward()
            
            nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / num_frames, total_accuracy / num_frames

def evaluate(placement_clstm, selection_rnn, data_iter, p_criterion, s_criterion,
             device, writer, curr_validation):
    total_p_loss = 0
    total_s_loss = 0
    total_p_acc = 0
    total_s_acc = 0

    with torch.no_grad():
       for i, batch in enumerate(data_iter):
            p_loss, p_acc, hiddens = run_placement_batch(placement_clstm, None,
                p_criterion, batch, device, writer, do_train=False)

            total_p_loss += p_loss
            total_p_acc += p_acc

            s_loss, s_acc = run_selection_batch(selection_rnn, None, s_criterion,
                batch, device, hiddens, do_train=False)
            
            total_s_loss += s_loss
            total_s_acc += s_acc

            if curr_validation >= 0:
                step = curr_validation * len(data_iter) + i
                writer.add_scalar('loss/valid_placement', p_loss, step)
                writer.add_scalar('accuracy/valid_placement', p_acc, step)
                writer.add_scalar('loss/valid_selection', s_loss, step)
                writer.add_scalar('accuracy/valid_selection', s_acc, step)

    return (total_p_loss / len(data_iter), total_p_acc / len(data_iter),
            total_s_loss / len(data_iter), total_s_acc / len(data_iter))

# full training process from placement -> selection
def run_models(train_iter, valid_iter, test_iter, num_epochs, device, save_dir, load_checkpoint, retrain, 
               dataset, early_stopping=True, print_every_x_epoch=1, validate_every_x_epoch=5):
    # setup or load models, optimizers
    placement_clstm = PlacementCLSTM(PLACEMENT_CHANNELS, PLACEMENT_FILTERS, PLACEMENT_KERNEL_SIZES,
                                     PLACEMENT_POOL_KERNEL, PLACEMENT_POOL_STRIDE, NUM_PLACEMENT_LSTM_LAYERS,
                                     PLACEMENT_INPUT_SIZE, HIDDEN_SIZE).to(device)
    placement_optim = optim.Adam(placement_clstm.parameters(), lr=PLACEMENT_LR)
    #placement_optim = optim.SGD(placement_clstm.parameters(), lr=PLACEMENT_LR, momentum=0.9)

    selection_rnn = SelectionRNN(NUM_SELECTION_LSTM_LAYERS, SELECTION_INPUT_SIZES[dataset.chart_type], 
                                 dataset.vocab_size, HIDDEN_SIZE, SELECTION_HIDDEN_WEIGHT).to(device)
    selection_optim = optim.Adam(selection_rnn.parameters(), lr=SELECTION_LR)
    #selection_optim = optim.SGD(selection_rnn.parameters(), lr=SELECTION_LR)
    
    # load model, optimizer states if resuming training
    best_placement_valid_loss = float('inf')
    best_selection_valid_loss = float('inf')
    start_epoch = 0
    start_epoch_batch = 0
    train_clstm = True
    train_srnn = True
    sub_logdir = datetime.datetime.now().strftime('%m_%d_%y_%H_%M')
  
    if load_checkpoint:
        checkpoint = load_save(load_checkpoint, retrain, placement_clstm, selection_rnn, device)
        if checkpoint:
            (start_epoch, start_epoch_batch, best_placement_valid_loss, 
             best_selection_valid_loss, train_clstm, train_srnn, sub_logdir) = checkpoint

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs', sub_logdir))

    print('Starting training..')
    for epoch in trange(num_epochs):
        if epoch < start_epoch:
            # enumerate data loaders to maintain consistent epoch batching
            for i, _ in enumerate(train_iter):
                pass
            
            if epoch % validate_every_x_epoch == 0:
                for i, _ in enumerate(valid_iter):
                    pass

            continue

        print('Epoch: {}'.format(epoch))
        epoch_p_loss = 0
        epoch_s_loss = 0
        # report_memory(device=device, show_tensors=True)

        for i, batch in enumerate(tqdm(train_iter)):
            # if resuming from checkpoint, skip batches until starting batch for the epoch
            if start_epoch_batch > 0:
                if i + 1 == start_epoch_batch:
                    start_epoch_batch = 0
                continue

            with torch.set_grad_enabled(train_clstm):
                placement_loss, placement_acc, clstm_hiddens = run_placement_batch(placement_clstm, 
                    placement_optim, PLACEMENT_CRITERION, batch, device, writer,
                    do_train=train_clstm, curr_train_epoch=epoch)

            with torch.set_grad_enabled(train_srnn):
                selection_loss, selection_acc = run_selection_batch(selection_rnn, selection_optim,
                    SELECTION_CRITERION, batch, device, clstm_hiddens, do_train=train_srnn)

            epoch_p_loss += placement_loss
            epoch_s_loss += selection_loss

            step = epoch * len(train_iter) + i
            writer.add_scalar('loss/train_placement', placement_loss, step)
            writer.add_scalar('accuracy/train_placement', placement_acc, step)
            writer.add_scalar('loss/train_selection', selection_loss, step)
            writer.add_scalar('accuracy/train_selection', selection_acc, step)

            if train_clstm:
                save_model(placement_clstm, save_dir, CLSTM_SAVE)
            if train_srnn:
                save_model(selection_rnn, save_dir, SRNN_SAVE)       

            save_checkpoint(epoch, i, best_placement_valid_loss,
                            best_selection_valid_loss, train_clstm, train_srnn, save_dir)

        epoch_p_loss = epoch_p_loss / len(train_iter)
        epoch_s_loss = epoch_s_loss / len(train_iter)

        if epoch % print_every_x_epoch == 0:
            print(f'\tAvg. training placement loss per unrolling: {epoch_p_loss:.5f}')
            print(f'\tAvg. training selection loss per frame: {epoch_s_loss:.5f}')

        if epoch % validate_every_x_epoch == 0:
            (placement_valid_loss, placement_valid_acc, selection_valid_loss,
             selection_valid_acc) = evaluate(placement_clstm, selection_rnn, valid_iter, 
                                             PLACEMENT_CRITERION, SELECTION_CRITERION, device, 
                                             writer, epoch / validate_every_x_epoch)
                
            print(f'\tAvg. validation placement loss per frame: {placement_valid_loss:.5f}')
            print(f'\tAvg. validation selection loss per frame: {selection_valid_loss:.5f}')

            # track best performing model(s) or save every epoch
            if early_stopping:
                better_placement = placement_valid_loss < best_placement_valid_loss
                better_selection = selection_valid_loss < best_selection_valid_loss

                if better_placement:
                    best_placement_valid_loss = placement_valid_loss
                    save_model(placement_clstm, save_dir, CLSTM_SAVE)
                else:
                    print("Placement validation loss increased, stopping CLSTM training")
                    train_clstm = False

                if better_selection:
                    best_selection_valid_loss = selection_valid_loss
                    save_model(selection_rnn, save_dir, SRNN_SAVE)
                else:
                    # cease training the placement model
                    print("Placement validation loss increased, stopping SRNN training")
                    train_srnn = False

                if not better_placement and not better_selection:
                    print("Both validation losses have increased. Stopping early..")
                    break

            save_checkpoint(epoch + 1, 0, best_placement_valid_loss, best_selection_valid_loss, 
                            train_clstm, train_srnn, save_dir)

    # evaluate on test set
    (placement_test_loss, placement_test_acc,
     selection_test_loss, selection_test_acc) = evaluate(placement_clstm, selection_rnn, test_iter, 
                                                         PLACEMENT_CRITERION, SELECTION_CRITERION,
                                                         device, writer, -1)

    # save training summary stats to json file
    summary_json = {
        'epochs_trained': num_epochs,
        'train_examples': len(train_iter.dataset),
        'valid_examples': len(valid_iter.dataset),
        'test_examples': len(test_iter.dataset),
        'placement_test_loss': placement_test_loss,
        'placement_test_accuracy': placement_test_acc,
        'selection_test_loss': selection_test_loss,
        'selection_test_accuracy': selection_test_acc
    }

    model_summary = log_training_stats(writer, dataset, summary_json)

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        f.write(json.dumps(model_summary, indent=2))

    # save special tokens for dataset vocabulary if needed
    if dataset.special_tokens:
        with open(os.path.join(save_dir, 'special_tokens.json'), 'w') as f:
            f.write(json.dumps(dataset.special_tokens, indent=2))

def log_training_stats(writer, dataset, summary_json):
    summary_json = {
        'total_charts': len(dataset),
        'unique_charts': dataset.n_unique_charts,
        'unique_songs': dataset.n_unique_songs,
        'audio_hours': dataset.n_audio_hours,
        'total_steps': dataset.n_steps,
        'min_level': dataset.min_level,
        'max_level': dataset.max_level,
        'avg_steps_per_second': dataset.avg_steps_per_second,
        'vocab_size': dataset.vocab_size
        **summary_json
    }

    # add other dataset text values
    writer.add_text('dataset_name', dataset.name)
    writer.add_text('chart_type', dataset.chart_type)
    writer.add_text('song_types', ', '.join(dataset.songtypes))
    writer.add_text('step_artists', ', '.join(dataset.step_artists))
    writer.add_text('permutations', ', '.join(dataset.permutations))

    hparam_dict = {
        'placement_lr': PLACEMENT_LR,
        'selection_lr': SELECTION_LR,
        'batch_size': BATCH_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'placement_unroll': PLACEMENT_UNROLLING_LEN,
        'selection_unroll': SELECTION_UNROLLING_LEN,
        'selection_hidden_wt': SELECTION_HIDDEN_WEIGHT
    }

    writer.add_hparams(hparam_dict, summary_json)
    writer.close()

    hparam_dict = {
        'placement_channels': PLACEMENT_CHANNELS,
        'placement_filters': PLACEMENT_FILTERS,
        'placement_kernels': PLACEMENT_KERNEL_SIZES,
        'placement_pool_kernel': PLACEMENT_POOL_KERNEL,
        'placement_pool_stride': PLACEMENT_POOL_STRIDE,
        'placement_lstm_layers': NUM_PLACEMENT_LSTM_LAYERS,
        'placement_input_size': PLACEMENT_INPUT_SIZE,
        'selection_lstm_layers': NUM_SELECTION_LSTM_LAYERS,
        'selection_input_size': SELECTION_INPUT_SIZES[dataset.chart_type],
        'type': 'rnns',
        'dataset_name': dataset.name,
        'chart_type': dataset.chart_type,
        'song_types': dataset.songtypes,
        'step_artists': dataset.step_artists,
        'permutations': dataset.permutations,
        **hparam_dict
    }

    return {**summary_json, **hparam_dict}

def get_dataloader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_charts,
                      shuffle=True)

def main():
    args = parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print('Device:', device)
    torch.manual_seed(SEED)

    # Retrieve/prepare data
    print('Loading dataset from {}...'.format(os.path.relpath(args.dataset_path)))
    dataset = StepchartDataset(args.dataset_path)

    train_data, valid_data, test_data = get_splits(dataset)
    train_iter = get_dataloader(train_data)
    valid_iter = get_dataloader(valid_data)
    test_iter = get_dataloader(test_data)

    datasets_size_str = (f'Total charts in dataset: {len(dataset)}\nTrain: {len(train_data)}, '
                         f'Valid: {len(valid_data)}, Test: {len(test_data)}')
    print(datasets_size_str)

    run_models(train_iter, valid_iter, test_iter, NUM_EPOCHS, device, args.save_dir,
               args.load_checkpoint, args.retrain, dataset)

if __name__ == '__main__':
    main()
