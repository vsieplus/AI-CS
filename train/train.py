# train a pair of models on the same dataset to perform step placement/selection

import os
import argparse
from pathlib import Path

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
    parser.add_argument('--output_dir', type=str, default=None,
        help='Specify custom output directory to save models to. If blank, will save in ./models/dataset_name')

    args = parser.parse_args()

    if not args.dataset_path and args.dataset_name:
        args.dataset_path = os.path.join(DATASETS_DIR, args.dataset_name)
    
    if not args.output_dir:
        args.output_dir = os.path.join(MODELS_DIR, os.path.split(args.dataset_path)[-1])

    return args

# train the placement models with the specified parameters
def train_placement_batch(cnn, rnn, optimizer, criterion, batch, device):
    cnn.train()
    rnn.train()
    optimizer.zero_grad()

    # [batch, 3, ?, 80] ; ? = (max batch) timesteps in audio features
    audio_feats = batch['audio_feats'].to(device)

    # [batch, # chart features]
    chart_feats = batch['chart_feats'].to(device)

    num_audio_frames = audio_feats.size(2)
    
    # both are [batch, # of chart frames]
    chart_placements = batch['placement_targets']
    chart_frames = batch.step_frames   # which audio frames recorded in chart data
    
    first_frame = chart_frames[0]
    last_frame = chart_frames[-1]

    total_loss = 0
    clstm_hiddens = []

    # bptt https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/29
    # unrolling https://machinelearningmastery.com/rnn-unrolling/
    # https://en.wikipedia.org/wiki/Backpropagation_through_time
    for frame in range(num_audio_frames):
        if frame < first_frame or frame > last_frame:
            continue

        cnn_in = audio_feats[:, :, ]

        cnn_out = cnn(cnn_in)
        logits, clstm_hidden = rnn(cnn_out, chart_feats)

        loss = criterion(logits, batch.chart_placements[frame])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        clstm_hiddens.append(clstm_hidden)

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
    print_every_x_batch=10, validate_every_x_epoch=5):

    cpu_device = torch.device('cpu')

    # setup models, optimizers
    placement_cnn = PlacementCNN(PLACEMENT_CHANNELS, PLACEMENT_FILTERS, PLACEMENT_KERNEL_SIZES)
    placement_rnn = PlacementRNN(NUM_PLACEMENT_LSTM_LAYERS, NUM_PLACEMENT_FEATURES, PLACEMENT_HIDDEN_SIZE)
    placement_optim = optim.Adam(list(cnn.parameters()) + list(rnn.parameters()), lr=PLACEMENT_LR)

    selection_rnn = SelectionRNN(NUM_SELECTION_LSTM_LAYERS, SELECTION_INPUT_SIZES[dataset_type], 
        HIDDEN_SIZE, SELECTION_HIDDEN_WEIGHT)
    selection_optim = optim.SGD(selection_rnn.parameters(), lr=SELECTION_LR)

    best_selection_valid_loss = float('inf')
    best_placement_valid_loss = float('inf')

    print('Starting training..')
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))

        for i, batch in enumerate(train_iter):
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

    # Retrieve/prepare data
    print('Loading dataset from {}...'.format(os.path.relpath(args.dataset_path)))
    dataset = StepchartDataset(args.dataset_path)
    dataset_type = dataset.chart_type
    
    train_data, valid_data, _ = get_splits(dataset)
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)

    run_models(train_iter, valid_iter, NUM_EPOCHS, dataset_type, args.output_dir, device)

if __name__ == '__main__':
    main()