# train a pair of models on the same dataset to perform step placement/selection

import os
import argparse
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hyper import *
from placement_model import PlacementCNN, PlacementRNN
from selection_model import SelectionRNN
from stepchart import StepchartDataset, get_splits, collate_charts

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset under data/datasets/subsets/')
    parser.add_argument('--dataset_path', type=str, default=None, help='Alternatively, provide direct path to dataset')
    parser.add_argument('--save_dir', type=str, default=None,
        help='Specify custom output directory to save models to. If blank, will save in ./models/dataset_name')

    args = parser.parse_args()

    if not args.dataset_path and args.dataset_name:
        args.dataset_path = os.path.join(DATASETS_DIR, args.dataset_name)
    
    if not args.save_dir:
        args.save_dir = os.path.join(MODELS_DIR, os.path.split(args.dataset_path)[-1])

    return args

# train the placement models with the specified parameters
def train_placement_batch(cnn, rnn, optimizer, criterion, batch, device):
    cnn.train()
    rnn.train()

    # [batch, 3, ?, 80] ; ? = (batch maximum) # of 10ms timesteps/frames in audio features
    audio_feats = batch['audio_feats'].to(device)
    num_audio_frames = audio_feats.size(2)
    batch_size = audio_feats.size(0)
    
    num_unrollings = PLACEMENT_UNROLLING_LEN // num_audio_frames + 1
    last_unroll_len = PLACEMENT_UNROLLING_LEN % num_audio_frames

    # [batch, # chart features]
    chart_feats = batch['chart_feats'].to(device)
    
    # both are [batch, max # of chart frames]
    chart_placements = batch['placement_targets'].to(device)    # if chart frame had hit or not
    step_frames = batch['step_frames']  # which audio frames recorded in chart data
    
    first_frame = step_frames[0]
    last_frame = step_frames[-1]

    total_loss = 0
    
    # final_shape -> [batch, # of nonempty chart frames, hidden]
    clstm_hiddens = [[] for _ in batch_size]

    # 100 steps of unrolling
    states = rnn.initStates(batch_size, device)
    for unrolling in range(num_unrollings):
        optimizer.zero_grad()

        # skip audio frames outside chart range
        audio_start_frame = max(unrolling * PLACEMENT_UNROLLING_LEN, first_frame)
        if unrolling = num_unrollings - 1:
            audio_end_frame = audio_start_frame + last_unroll_len
        else:         
            audio_end_frame = audio_start_frame + PLACEMENT_UNROLLING_LEN
        audio_end_frame = min(audio_end_frame, last_frame)
        unroll_length = audio_end_frame - audio_start_frame
            
        # [batch, <=100, 83] (batch, unroll length, audio features)
        cnn_output = cnn(audio_feats[:, :, audio_start_frame:audio_end_frame])

        # [batch, unroll_len] / [batch, unroll_len, hidden] / (*...hidden, *...hidden)
        logits, clstm_hidden, states = rnn(cnn_output, chart_feats, states)

        # for each batch example, if this audio frame also in the chart, use actual target at that frame
        targets = torch.zeros_like(logits.size, device=device)
        for b in range(batch_size):
            for frame in range(unroll_length):
                absolute_frame = audio_start_frame + frame
                frame_idx = 0
                try:
                    frame_idx =  step_frames[b].index(absolute_frame, start=frame_idx):
                    targets[b, frame] = chart_placements[b, frame_idx]

                    # only need to track chart frame hiddens with nonempty steps
                    clstm_hiddens[b].append(clstm_hidden[b, frame])
                except ValueError:
                    # else there was no step
                    targets[b, frame] = 0
                
        # compute total loss for this unrolling
        loss = criterion(logits, targets)
        loss.backward()

        # clip grads before step if l2 norm > 5
        nn.utils.clip_grad_norm_(optimizer.params, max_norm=PLACEMENT_MAX_NORM, type=2)
        optimizer.step()

        total_loss += loss.item()

    return total_loss, clstm_hiddens
   
def train_selection_batch(rnn, optimizer, criterion, batch, device, clstm_hiddens=None):
    rnn.train()

def evaluate(p_cnn, p_rnn, s_rnn, valid_iter, p_criterion, s_criterion, device):
    s_rnn.eval()

    total_placement_loss = 0

    with torch.no_grad():
        for batch in valid_iter:
            total_placement_loss += evaluate_placement_batch(p_cnn, p_rnn, batch)

    return total_loss

def evaluate_placement_batch(cnn, rnn, criterion, batch):
    cnn.eval()
    rnn.eval()
    

def evaluate_selection_batch(rnn, criterion, batch, clstm_hiddens):
    rnn.eval()

# full training process from placement -> selection
def run_models(train_iter, valid_iter, num_epochs, dataset_type, device, save_dir, early_stopping=True,
    print_every_x_batch=10, validate_every_x_epoch=2):

    cpu_device = torch.device('cpu')

    # setup models, optimizers
    placement_cnn = PlacementCNN(PLACEMENT_CHANNELS, PLACEMENT_FILTERS, PLACEMENT_KERNEL_SIZES
        PLACEMENT_POOL_KERNEL, PLACEMENT_POOL_STRIDE).to(device)
    placement_rnn = PlacementRNN(NUM_PLACEMENT_LSTM_LAYERS, PLACEMENT_INPUT_SIZE, PLACEMENT_HIDDEN_SIZE).to(device)
    placement_optim = optim.SGD(list(cnn.parameters()) + list(rnn.parameters()), 
        lr=PLACEMENT_LR, momentum=0.9)

    selection_rnn = SelectionRNN(NUM_SELECTION_LSTM_LAYERS, SELECTION_INPUT_SIZES[dataset_type], 
        HIDDEN_SIZE, SELECTION_HIDDEN_WEIGHT).to(device)
    selection_optim = optim.SGD(selection_rnn.parameters(), lr=SELECTION_LR, momentum=0.9)

    best_selection_valid_loss = float('inf')
    best_placement_valid_loss = float('inf')

    print('Starting training..')
    for epoch in tqdm(range(num_epochs)):
        print('Epoch: {}'.format(epoch))

        for i, batch in tqdm(enumerate(train_iter)):
            placement_loss, clstm_hiddens = train_placement_batch(placement_cnn,
                placement_rnn, placement_optim, PLACEMENT_CRITERION, batch, device)

            selection_loss = train_selection_batch(selection_rnn, selection_optim,
                SELECTION_CRITERION, batch, device, clstm_hiddens)

            if (i + 1) % print_every_x_batch:
                print(f'Batch {i}')
                print(f'Placement model batch loss: {placement_loss}')        
                print(f'Placement model batch loss: {selection_loss}')       

        if (epoch + 1) % validate_every_x_epoch == 0:
            placement_valid_loss = evaluate(placement_cnn, placement_rnn, selection_rnn,
                valid_iter, PLACEMENT_CRITERION, SELECTION_CRITERION, device)

            if placement_valid_loss < best_placement_valid_loss:
                best_placement_valid_loss = placement_valid_loss
                save_model((placement_cnn, placement_rnn), save_dir)
            else:
                break # early stopping

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

    print(f'Total charts in dataset: {len(dataset)}; Train: {len(train_data)}, 
        Valid: {len(valid_data)}, Test: {len(test_data)}')

    run_models(train_iter, valid_iter, NUM_EPOCHS, dataset_type, args.save_dir, device)

if __name__ == '__main__':
    main()